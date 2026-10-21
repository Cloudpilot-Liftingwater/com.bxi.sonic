"""Official PICO button replay and guarded service upgrade; no robot or ROS I/O."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import test_buttons as buttons
try:
    import install_service as installer
except ModuleNotFoundError as exc:
    if exc.name != "install_service":
        raise
    installer = None  # Historical field installer is excluded from customer distributions.
import trial_lifecycle as life


def manager(t, toggle=False, discard=False, mode=1, calibrated=True):
    return buttons.PicoSnapshot(dict(
        stream_mode=np.array([mode]), calibration_ready=np.array([calibrated]),
        toggle_data_collection=np.array([toggle]), toggle_data_abort=np.array([discard])), t)


class OfficialButtons(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.child = mock.Mock(pid=123, returncode=None, poll=mock.Mock(return_value=None))
        self.popen = mock.Mock(return_value=self.child)
        self.s = buttons.Supervisor(self.root, self.root/'status.json', self.popen)
        self.gate = buttons.ButtonGate(source_clock=False)

    def tearDown(self):
        if self.s.log: self.s.log.close()
        self.temp.cleanup()

    def event(self, p, *, pose=True, age=.01):
        self.s.manager_event(p, self.gate, p.received_monotonic+age, 1000.,
                             p.received_monotonic if pose else None)

    def test_release_start_save_and_new_episode(self):
        self.event(manager(1, toggle=True))
        self.popen.assert_not_called()
        self.event(manager(1.1)); self.event(manager(1.2, toggle=True))
        self.assertEqual(buttons.read_intent(self.s.folder)['command'], 'start')
        first = self.s.folder
        self.event(manager(1.3, toggle=True))  # held
        self.assertEqual(buttons.read_intent(first)['sequence'], 1)
        self.event(manager(1.4)); self.event(manager(1.5, toggle=True))
        self.assertEqual(buttons.read_intent(first)['command'], 'finish')
        self.event(manager(1.6)); self.event(manager(1.7, toggle=True))
        self.assertEqual(self.popen.call_count, 1)  # saving not interrupted
        self.child.returncode = 0; self.child.poll.return_value = 0; self.s.poll()
        self.child.returncode = None; self.child.poll.return_value = None
        self.event(manager(1.8)); self.event(manager(1.9, toggle=True))
        self.assertEqual(self.popen.call_count, 2)
        self.assertNotEqual(self.s.folder, first)

    def test_discard_wins_over_toggle_and_never_restarts(self):
        self.s.event('start')
        self.event(manager(1)); self.event(manager(1.1, toggle=True, discard=True))
        self.assertEqual(buttons.read_intent(self.s.folder)['command'], 'discard')
        self.s.send('abort', 'lost_stream')
        self.event(manager(1.2)); self.event(manager(1.3, toggle=True))
        self.assertEqual(buttons.read_intent(self.s.folder)['command'], 'discard')
        self.assertEqual(self.popen.call_count, 1)

    def test_discard_idle_is_noop_and_held_after_reconnect_rejected(self):
        self.event(manager(1)); self.event(manager(1.1, discard=True))
        self.event(manager(3, toggle=True))
        self.popen.assert_not_called()
        self.event(manager(3.1)); self.event(manager(3.2, toggle=True))
        self.popen.assert_called_once()

    def test_official_start_requires_fresh_pose_and_mode_and_calibration(self):
        for kwargs in (dict(mode=0), dict(calibrated=False), {}):
            self.gate = buttons.ButtonGate(source_clock=False)
            self.event(manager(1))
            self.event(manager(1.1, toggle=True, **kwargs), pose=bool(kwargs))
        self.popen.assert_not_called()
        self.event(manager(1.2))
        self.event(manager(1.3, toggle=True), age=.6)
        self.popen.assert_not_called()

    def test_stale_pose_cannot_start_but_active_episode_can_save(self):
        self.event(manager(1))
        self.s.manager_event(manager(1.1, toggle=True), self.gate, 1.11, 1000., .1)
        self.popen.assert_not_called()
        self.s.event('start')
        self.event(manager(1.2)); self.event(manager(1.3, toggle=True), pose=False)
        self.assertEqual(buttons.read_intent(self.s.folder)['command'], 'finish')

    def test_pose_duplicate_official_toggle_cannot_toggle_twice(self):
        gate = buttons.ButtonGate()
        gate.accept(buttons.packet(1), 1.01, 101.01)
        packet = buttons.packet(1.1)
        packet.fields['toggle_data_collection'] = np.array([True])
        self.assertIsNone(gate.accept(packet, 1.11, 101.11))

    def test_malformed_official_toggle_rejected(self):
        self.event(manager(1))
        p = manager(1.1, toggle=True)
        p.fields['toggle_data_collection'] = np.array([float('nan')])
        self.event(p)
        self.popen.assert_not_called()

    def test_busy_pending_start_toggle_cancels_instead_of_late_recording(self):
        self.event(manager(1)); self.event(manager(1.1, toggle=True))
        self.event(manager(1.2)); self.event(manager(1.3, toggle=True))
        self.assertEqual(buttons.read_intent(self.s.folder)['command'], 'finish')
        self.assertEqual(self.s.phase, 'SAVING')

    def test_truncated_child_report_does_not_kill_or_restart_listener(self):
        self.s.event('start')
        (self.s.folder/'launcher_status.json').write_text('{')
        self.child.returncode = 0; self.child.poll.return_value = 0
        self.s.poll(); self.s.publish_status()
        self.assertIn('ERROR', self.s.phase)
        self.assertIn('report_error', self.s.last_result)
        self.assertEqual(self.popen.call_count, 1)

    def test_bad_status_does_not_claim_recording_or_crash(self):
        self.s.event('start')
        (self.s.folder/'session_status.json').write_text('{')
        self.s.publish_status()
        value = json.loads((self.root/'status.json').read_text())
        self.assertIn('collector_status_error', value)
        self.assertEqual(value['phase'], 'STARTING')


class DiscardAudit(unittest.TestCase):
    def test_discard_without_rows_is_operator_success_not_training_pass(self):
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temp)
            report = life.audit_after_stop(root/'episode_0000', root, None,
                                           dict(outcome='discarded', reason='operator'))
            self.assertEqual(report['exit_code'], 0)
            self.assertEqual(report['outcome'], 'discarded')
            self.assertTrue(report['no_saved_episode'])
            self.assertTrue(report['excluded_from_training'])

    def test_discarded_rows_audited_and_concurrent_camera_failure_not_hidden(self):
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temp); path = root/'_discarded'/'episode_0000_test'
            path.mkdir(parents=True); (path/'meta.json').write_text('{}')
            result = dict(readback='PASS', discarded=True, episode_complete=False)
            with mock.patch.object(life, 'verify', return_value=result) as verify:
                report = life.audit_after_stop(root/'episode_0000', root, None)
                self.assertEqual(report['exit_code'], 0)
                verify.assert_called_once_with(path, allow_incomplete=True)
                report = life.audit_after_stop(root/'episode_0000', root, dict(reason='usb_disconnect'))
                self.assertEqual(report['exit_code'], 4)

    def test_bad_discarded_image_cannot_be_hidden_by_operator_outcome(self):
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temp); path = root/'_discarded'/'episode_0000_test'
            path.mkdir(parents=True); (path/'meta.json').write_text('{}')
            with mock.patch.object(life, 'verify', side_effect=ValueError('bad JPEG')):
                report = life.audit_after_stop(root/'episode_0000', root, None,
                                               dict(outcome='discarded'))
            self.assertEqual(report['exit_code'], 4)
            self.assertTrue(path.is_dir())


@unittest.skipIf(installer is None, "historical installer excluded from customer package")
class SafeInstallation(unittest.TestCase):
    def test_known_old_unit_only_replaced_with_backup_and_opt_in(self):
        previous = installer.ROOT.parent/'sonic_pico_collection_20261008_r3'/installer.NAME
        if not previous.exists(): self.skipTest('development baseline only')
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); target = root/installer.NAME
            target.write_bytes(previous.read_bytes())
            with self.assertRaises(RuntimeError): installer.install_unit(installer.ROOT/installer.NAME, target)
            installer.install_unit(installer.ROOT/installer.NAME, target, replace_known=True)
            backup = list(root.glob('elf3_pico_collection_backup_*/*.service'))
            self.assertEqual(len(backup), 1)
            self.assertEqual(backup[0].read_bytes(), previous.read_bytes())
            self.assertEqual(target.read_bytes(), (installer.ROOT/installer.NAME).read_bytes())

    def test_unrecognized_unit_not_overwritten_even_with_replace_known(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/installer.NAME; path.write_text('unrelated unit')
            with self.assertRaises(RuntimeError):
                installer.install_unit(installer.ROOT/installer.NAME, path, replace_known=True)
            self.assertEqual(path.read_text(), 'unrelated unit')

    def test_idle_check_refuses_recording_starting_saving_and_stale_status(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'status.json'
            for phase in ('RECORDING', 'STARTING', 'WAITING_FOR_INPUTS', 'SAVING', 'AUDITING', 'DISCARDING'):
                path.write_text(json.dumps(dict(phase=phase, updated_unix=100.)))
                with self.assertRaises(RuntimeError): installer.require_idle_status(path, 101.)
            path.write_text(json.dumps(dict(phase='WAITING_FOR_BUTTON', updated_unix=100.)))
            installer.require_idle_status(path, 101.)
            with self.assertRaises(RuntimeError): installer.require_idle_status(path, 110.)
            with self.assertRaises(RuntimeError): installer.require_idle_status(path, 101., started_after=100.)


if __name__ == '__main__': unittest.main()
