"""Offline button/ownership tests; subprocesses are mocked, no DDS or hardware."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'collector')]
import numpy as np
from real_robot_collection.pico import PicoSnapshot
from session_protocol import ButtonGate, atomic_json, intent, read_intent, wait_for_session
from button_supervisor import Supervisor


def packet(t, start=False, finish=False, abort=False, mode=1, calibrated=True, source=None):
    fields = dict(timestamp_realtime=[t+100 if source is None else source],
                  toggle_data_collection=[start], finish_data_collection=[finish],
                  toggle_data_abort=[abort], stream_mode=[mode], calibration_ready=[calibrated])
    return PicoSnapshot({k:np.array(v) for k,v in fields.items()}, t)


class GateTests(unittest.TestCase):
    def setUp(self):
        self.gate = ButtonGate(source_clock=False)

    def accept(self, p):
        return self.gate.accept(p, p.received_monotonic+.01, p.received_monotonic+100.01)

    def test_initial_held_press_is_not_automatic_recording(self):
        self.assertIsNone(self.accept(packet(1, start=True)))
        self.assertIsNone(self.accept(packet(1.1)))
        self.assertEqual(self.accept(packet(1.2, start=True)), 'toggle')

    def test_held_and_duplicate_packets_launch_once(self):
        self.accept(packet(1))
        self.assertEqual(self.accept(packet(1.1, start=True)), 'toggle')
        self.assertIsNone(self.accept(packet(1.2, start=True)))
        self.assertIsNone(self.accept(packet(1.2, start=True)))

    def test_release_then_new_press_is_new_episode(self):
        self.accept(packet(1)); self.accept(packet(1.1, start=True))
        self.accept(packet(1.2))
        self.assertEqual(self.accept(packet(1.3, start=True)), 'toggle')

    def test_discard_wins_over_toggle(self):
        self.accept(packet(1))
        self.assertEqual(self.accept(packet(1.1, start=True, abort=True)), 'discard')

    def test_pose_and_legacy_aliases_never_control_recording(self):
        self.gate = ButtonGate()
        self.accept(packet(1))
        p = packet(1.1, start=True, abort=True)
        p.fields.update(start_data_collection=np.array([True]), finish_data_collection=np.array([True]))
        self.assertIsNone(self.accept(p))

    def test_stale_or_future_source_or_receive_rejected(self):
        self.gate = ButtonGate()
        for p, now, wall in ((packet(1, start=True), 2, 101),
                             (packet(1, start=True, source=102), 1.01, 101.01),
                             (packet(1, start=True, source=float('nan')),1.01,101.01)):
            self.accept(packet(.9))
            self.assertIsNone(self.gate.accept(p, now, wall))

    def test_disconnect_requires_release_before_new_press(self):
        self.accept(packet(1))
        self.assertIsNone(self.accept(packet(3, start=True)))
        self.accept(packet(3.1))
        self.assertEqual(self.accept(packet(3.2, start=True)), 'toggle')

    def test_reordered_source_cannot_trigger(self):
        self.gate = ButtonGate()
        self.accept(packet(1))
        self.assertIsNone(self.accept(packet(1.1, start=True, source=100.9)))

    def test_malformed_numeric_flags_do_not_crash_listener_or_start(self):
        self.accept(packet(1))
        for field in ('stream_mode', 'toggle_data_collection', 'calibration_ready'):
            p = packet(1.1, start=True)
            p.fields[field] = np.array([float('nan')])
            self.assertIsNone(self.accept(p))


class SupervisorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir='/tmp')
        self.root = Path(self.temp.name)
        self.child = Mock(pid=123, poll=Mock(return_value=None), returncode=None)
        self.popen = Mock(return_value=self.child)
        self.s = Supervisor(self.root, self.root/'status.json', self.popen)

    def tearDown(self):
        if self.s.log:
            self.s.log.close()
        self.temp.cleanup()

    def test_idle_never_launches_cameras_or_controller(self):
        self.s.poll(); self.s.publish_status()
        self.popen.assert_not_called()
        self.assertEqual(json.loads((self.root/'status.json').read_text())['phase'], 'WAITING_FOR_BUTTON')

    def test_start_has_fresh_private_folder_and_only_collection_command(self):
        self.s.event('start')
        cmd = self.popen.call_args.args[0]
        self.assertEqual(Path(cmd[1]).name, 'start.sh')
        self.assertEqual(cmd[2], '--session-dir')
        self.assertTrue(self.popen.call_args.kwargs['start_new_session'])
        self.assertEqual(read_intent(self.s.folder)['command'], 'start')
        self.assertEqual(self.s.folder.stat().st_mode & 0o077, 0)

    def test_duplicate_start_busy_including_finalization_does_not_launch(self):
        self.s.event('start'); self.s.event('start'); self.s.event('finish'); self.s.event('start')
        self.assertEqual(self.popen.call_count, 1)
        self.assertEqual(read_intent(self.s.folder)['command'], 'finish')

    def test_early_finish_replaces_unconsumed_start_and_never_auto_restarts(self):
        self.s.event('start'); self.s.event('finish')
        request = read_intent(self.s.folder)
        self.assertEqual(request['sequence'], 2)
        self.assertEqual(request['command'], 'finish')
        self.s.send('start', 'late')
        self.assertEqual(read_intent(self.s.folder), request)

    def test_child_failure_retained_and_next_explicit_start_uses_new_folder(self):
        self.s.event('start'); first = self.s.folder
        self.child.poll.return_value = 4; self.child.returncode = 4
        self.s.poll()
        self.assertIn('ERROR', self.s.phase)
        self.assertEqual(self.popen.call_count, 1)
        self.child.poll.return_value = None
        self.s.event('start')
        self.assertNotEqual(self.s.folder, first)
        self.assertTrue((first/'intent.json').exists())

    def test_save_timing_review_returns_to_waiting(self):
        self.s.event('start')
        self.child.poll.return_value = 2; self.child.returncode = 2
        self.s.poll()
        self.assertEqual(self.s.phase, 'WAITING_FOR_BUTTON')

    def test_journal_status_reflects_actual_recording_not_just_process_start(self):
        self.s.event('start')
        atomic_json(self.s.folder/'session_status.json', dict(recording=True, accepted_rows=23,
                    missing=[], button_session=dict(pending_start=False, done=False)))
        self.s.publish_status()
        value = json.loads((self.root/'status.json').read_text())
        self.assertEqual(value['phase'], 'RECORDING')
        self.assertEqual(value['collector']['accepted_rows'], 23)

    def test_ax_mode_exit_pauses_and_resumes_same_session(self):
        self.s.event('start')
        first = self.s.folder
        gate = ButtonGate(source_clock=False)
        p = PicoSnapshot(dict(stream_mode=np.array([2]), calibration_ready=np.array([True])), 10.)
        self.s.manager_event(p, gate, 10.01, 1000.)
        request = read_intent(self.s.folder)
        self.assertEqual(request['command'], 'pause')
        self.assertEqual(request['reason'], 'pico_left_pose_mode')
        self.s.input_health(10.5, 8.)
        self.assertEqual(read_intent(self.s.folder), request)
        self.s.manager_event(packet(15), gate, 15.01, 1000., live_pose_at=15.)
        request = read_intent(first)
        self.assertEqual(request['command'], 'resume')
        self.assertEqual(request['pauses'], [[10., 15.]])
        self.assertEqual(self.s.folder, first)
        self.assertEqual(self.popen.call_count, 1)
        self.s.event('finish')
        self.s.send('resume', 'late')
        self.assertEqual(read_intent(first)['command'], 'finish')

    def test_manager_state_cannot_start_without_source_pose(self):
        gate = ButtonGate(source_clock=False)
        self.s.manager_event(packet(1), gate, 1.01, 100.)
        self.s.manager_event(packet(1.1, start=True), gate, 1.11, 100.)
        self.popen.assert_not_called()

    def test_stale_manager_exit_does_not_stop_new_recording(self):
        self.s.event('start')
        gate = ButtonGate(source_clock=False)
        self.s.manager_event(packet(1, mode=2), gate, 2., 100.)
        self.assertEqual(read_intent(self.s.folder)['command'], 'start')

    def test_wrong_session_and_invalid_intent_rejected(self):
        self.s.event('start')
        for change in (dict(session='wrong'), dict(sequence=True), dict(command='enable'),
                       dict(monotonic='NaN')):
            request = intent(self.s.folder, 2, 'start', 'test'); request.update(change)
            atomic_json(self.s.folder/'intent.json', request)
            with self.assertRaises(ValueError): read_intent(self.s.folder)

    def test_shutdown_signals_only_owned_launcher(self):
        self.s.event('start')
        def waited(**_): self.child.poll.return_value = 0; self.child.returncode = 0
        self.child.wait.side_effect = waited
        with patch('button_supervisor.os.killpg') as kill:
            self.s.close()
        self.assertEqual(kill.call_args.args[0], 123)
        self.assertIsNone(self.s.child)

    def test_monitor_stops_on_no_episode_cancel(self):
        folder = self.root/'session'; folder.mkdir()
        atomic_json(folder/'session_result.json', dict(reason='start_cancelled'))
        result = wait_for_session(self.root/'episode_0000', self.root, [], folder)
        self.assertEqual(result['reason'], 'start_cancelled')

    def test_monitor_watchdog_is_bounded_even_if_node_is_stuck(self):
        atomic_json(self.root/'intent.json', intent(self.root, 1, 'start', 'test', now=0.))
        with patch('session_protocol.time.monotonic', side_effect=[0., 1000.]):
            result = wait_for_session(self.root/'episode_0000', self.root, [], self.root)
        self.assertEqual(result['reason'], 'session_watchdog_timeout')

    def test_deliberate_pause_requires_manager_heartbeat_and_fresh_resume(self):
        self.s.event('start')
        gate = ButtonGate(source_clock=False)
        self.s.manager_event(packet(10, mode=2), gate, 10.01, 1000.)
        self.s.manager_event(packet(10.5), gate, 10.51, 1000., live_pose_at=9.9)
        self.assertEqual(read_intent(self.s.folder)['command'], 'pause')
        self.s.input_health(10.6, 9.9)  # allow POSE restart time
        self.assertEqual(read_intent(self.s.folder)['command'], 'pause')
        self.s.input_health(11.6, 9.9)
        self.assertEqual(read_intent(self.s.folder)['command'], 'abort')

    def test_pose_loss_without_mode_pause_is_still_a_fault(self):
        self.s.event('start')
        self.s.input_health(12., 10.)
        self.assertEqual(read_intent(self.s.folder)['command'], 'abort')

    def test_pause_heartbeat_works_beyond_six_hundred_seconds(self):
        self.s.event('start')
        gate = ButtonGate(source_clock=False)
        for stamp in (10., 710., 1010.):
            self.s.manager_event(packet(stamp, mode=2), gate, stamp+.01, 2000.)
            self.s.input_health(stamp+.1, 9.)
            self.assertEqual(read_intent(self.s.folder)['command'], 'pause')
        self.assertEqual(read_intent(self.s.folder)['pauses'], [[10., None]])
        self.s.manager_event(packet(1010.2), gate, 1010.21, 2000., live_pose_at=1010.2)
        self.assertEqual(read_intent(self.s.folder)['command'], 'resume')
        self.assertEqual(self.popen.call_count, 1)

    def test_save_and_discard_win_over_paused_child_status(self):
        for command, phase in (('finish', 'SAVING'), ('discard', 'DISCARDING')):
            if self.s.log: self.s.log.close()
            self.s.child = None
            self.s.event('start')
            self.s.send('pause', 'ax')
            atomic_json(self.s.folder/'session_status.json', dict(recording=True,
                button_session=dict(paused=True, done=False)))
            self.s.event(command); self.s.publish_status()
            self.assertEqual(json.loads(self.s.status_file.read_text())['phase'], phase)

    def test_parent_watchdog_extends_across_pause_but_still_bounds_active_time(self):
        atomic_json(self.root/'intent.json', intent(self.root, 2, 'pause', 'test', now=10., pauses=[[10., None]]))
        def finish_pause(_):
            atomic_json(self.root/'intent.json', intent(self.root, 3, 'resume', 'test', now=1000., pauses=[[10., 1000.]]))
        with patch('session_protocol.time.monotonic', side_effect=[0., 1000., 2000.]), \
             patch('session_protocol.time.sleep', side_effect=finish_pause) as sleep:
            result = wait_for_session(self.root/'episode_0000', self.root, [], self.root)
        sleep.assert_called_once()
        self.assertEqual(result['reason'], 'session_watchdog_timeout')


if __name__ == '__main__':
    unittest.main()
