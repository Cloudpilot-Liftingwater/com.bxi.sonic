"""Automatic selection is preflight only; active recorders keep fixed contracts."""
import itertools
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

import test_identity as t
from auto_identity import AutoIdentityGate
from button_supervisor import Supervisor
from real_robot_collection.gripper_history import GripperHistory
from real_robot_collection.gripper_identity import identities


def sample(pair=('dm', 'bxi'), now=10.):
    return t.fixture_pair(*pair, stamp=now+90.)[1]['gripper'][0].raw


def prime(gate, pair=('dm', 'bxi'), now=10.):
    for delta in (0., .02):
        assert gate.accept(json.dumps(sample(pair, now+delta)), now+delta)


class AutoIdentityTests(unittest.TestCase):
    def test_all_four_topologies_resolve_to_explicit_child_arguments(self):
        for pair in itertools.product(('dm', 'bxi'), repeat=2):
            gate = AutoIdentityGate(); prime(gate, pair)
            expected = identities(*pair)
            self.assertEqual(gate.arguments(10.03, 1),
                ('--left', pair[0], '--right', pair[1], '--left-id', str(expected[0]['motor_id']),
                 '--right-id', str(expected[1]['motor_id']), '--left-bus', '5', '--right-bus', '6'))

    def test_zero_or_multiple_publishers_rejected(self):
        gate = AutoIdentityGate(); prime(gate)
        for count in (0, 2, 3):
            with self.assertRaisesRegex(ValueError, 'exactly one'): gate.arguments(10.03, count)

    def test_two_distinct_fresh_samples_required(self):
        gate = AutoIdentityGate()
        with self.assertRaises(ValueError): gate.arguments(10., 1)
        raw = sample()
        self.assertTrue(gate.accept(json.dumps(raw), 10.))
        with self.assertRaises(ValueError): gate.arguments(10., 1)
        self.assertFalse(gate.accept(json.dumps(raw), 10.01))
        with self.assertRaises(ValueError): gate.arguments(10.01, 1)
        prime(gate)
        with self.assertRaises(ValueError): gate.arguments(10.5, 1)

    def test_invalid_stale_future_and_nonhuman_samples_clear_cached_identity(self):
        for mutate in (lambda x:x.update(training_eligible=False),
                       lambda x:x.update(ineligible_reasons=['pose_paused']),
                       lambda x:x.update(monotonic=2.),
                       lambda x:x.update(monotonic=20.),
                       lambda x:x['samples'][0].update(profile='wrong'),
                       lambda x:x['samples'][0].update(position_unit='mm'),
                       lambda x:x['samples'][0].update(motor_id=8),
                       lambda x:x['samples'][0].update(bus=6)):
            gate = AutoIdentityGate(); prime(gate)
            raw = sample(now=10.04); mutate(raw)
            self.assertFalse(gate.accept(json.dumps(raw), 10.04))
            with self.assertRaises(ValueError): gate.arguments(10.05, 1)

    def test_protocol_change_needs_two_new_samples(self):
        gate = AutoIdentityGate(); prime(gate)
        gate.accept(json.dumps(sample(('bxi', 'bxi'), 10.04)), 10.04)
        with self.assertRaises(ValueError): gate.arguments(10.04, 1)
        gate.accept(json.dumps(sample(('bxi', 'bxi'), 10.06)), 10.06)
        self.assertEqual(gate.arguments(10.07, 1)[1:4], ('bxi', '--right', 'bxi'))

    def test_start_waits_for_identity_but_save_does_not(self):
        with tempfile.TemporaryDirectory() as temp:
            gate = AutoIdentityGate()
            child = Mock(poll=Mock(return_value=None))
            popen = Mock(return_value=child)
            s = Supervisor(temp, Path(temp)/'status.json', popen,
                           collection_args_provider=lambda:gate.arguments(10.03, 1))
            try:
                with self.assertRaises(ValueError): s.event('start')
                popen.assert_not_called()
                self.assertEqual(list(Path(temp).iterdir()), [])
                prime(gate)
                s.event('start')
                frozen = tuple(popen.call_args.args[0][-12:])
                self.assertEqual(frozen, gate.arguments(10.03, 1))
                gate.reject('lost telemetry')
                s.event('toggle')
                self.assertEqual(s.command, 'finish')
                self.assertEqual(s.active_collection_args, frozen)
                popen.assert_called_once()
            finally:
                if s.log: s.log.close()

    def test_next_episode_can_select_new_pair_without_relabeling_old(self):
        with tempfile.TemporaryDirectory() as temp:
            gate = AutoIdentityGate(); prime(gate)
            child = Mock(poll=Mock(return_value=None), returncode=0)
            popen = Mock(return_value=child)
            now = [10.03]
            s = Supervisor(temp, Path(temp)/'status.json', popen,
                           collection_args_provider=lambda:gate.arguments(now[0], 1))
            try:
                s.event('start'); old_args = s.active_collection_args
                child.poll.return_value = 0; s.poll()
                prime(gate, ('bxi', 'bxi'), now=20.); now[0] = 20.03
                child.poll.return_value = None; s.event('start')
                self.assertNotEqual(s.active_collection_args, old_args)
                self.assertEqual(old_args[1], 'dm')
                self.assertEqual(s.active_collection_args[1], 'bxi')
            finally:
                if s.log: s.log.close()

    def test_existing_collector_rejects_mid_episode_hardware_change(self):
        history = GripperHistory(expected=identities('dm', 'bxi'))
        samples, reason = history.ingest(json.dumps(sample(('bxi', 'bxi'))), 10.02, 90.)
        self.assertFalse(samples)
        self.assertIsNotNone(reason)

    def test_r4_unit_is_recognized_for_scoped_upgrade(self):
        import hashlib
        import install_service as installer
        old = Path(__file__).resolve().parents[2]/'sonic_pico_collection_20261008_r4/elf3_pico_collection.service'
        if old.exists(): self.assertIn(hashlib.sha256(old.read_bytes()).hexdigest(), installer.KNOWN_UNITS)
