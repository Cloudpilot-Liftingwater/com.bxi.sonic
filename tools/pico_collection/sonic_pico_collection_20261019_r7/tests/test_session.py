"""Session node tests with mocked ROS transport and synthetic saved images only."""
import importlib
import json
from pathlib import Path
import tempfile
import time
import sys
from types import SimpleNamespace
import unittest
from unittest import mock
import yaml

import test_trial as t
from session_protocol import atomic_json, intent, MAX_EPISODE_SECONDS


class SessionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        base = t.load_node()
        with mock.patch.dict(sys.modules, {
                'real_robot_collection.node': SimpleNamespace(RealCollectionNode=base.RealCollectionNode),
                'real_robot_collection.source_node': base}):
            cls.mod = importlib.import_module('session_node')

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.folder = Path(self.temp.name)
        cfg = t.configuration(); cfg['dataset']['output_root'] = str(self.folder/'data')
        # These legacy fixtures use 30 Hz camera anchors; the site default is now 50 Hz.
        cfg['metadata']['capture_profile'] = 'head30'
        cfg['metadata']['storage_backend'] = 'legacy_jpeg'
        cfg['dataset']['fps'] = 30
        path = self.folder/'config.yaml'; path.write_text(yaml.safe_dump(cfg))
        with mock.patch.dict(self.mod.RealCollectionNode.__init__.__globals__,
                             PicoSubscriber=mock.Mock(), smpl_subscriber=mock.Mock()):
            self.n = self.mod.SessionNode(path, 'SYNTHETIC_SESSION_ONLY', self.folder)
        self.start_time = time.monotonic()
        self.write(1, 'start')

    def tearDown(self):
        self.n.close()
        self.temp.cleanup()

    def write(self, seq, command, when=None, pauses=None):
        atomic_json(self.folder/'intent.json', intent(self.folder, seq, command, 'unit_test',
            self.start_time if when is None else when, self.n.pauses if pauses is None else pauses))

    def tick(self):
        self.n.last_control_poll = 0.
        with mock.patch.object(self.n, '_clock_ok', return_value=True), \
             mock.patch.object(self.mod.shutil, 'disk_usage', return_value=SimpleNamespace(free=5*1024**3)):
            self.n._control_tick()

    def fake_recording(self):
        self.n.recorder.start_episode(dict(task_id='TEST', instruction_text='SYNTHETIC ONLY'), runtime_metadata={})
        self.n.state_machine.start()
        self.n.latest_state_machine = t.JsonSample(time.monotonic(), time.time(),
                                                   {'current': {'name': self.n._required_state_name()}})
        self.n.finish_deadline = time.monotonic()+600
        self.n.capture_end_unix = time.time()+600

    def test_only_recording_status_publisher(self):
        self.assertEqual(self.n.publications, ['/elf3/recording_status'])

    def test_initial_start_waits_for_ready_then_starts_once(self):
        with mock.patch.object(self.n, '_ready', return_value=['head_rgb']) as ready, \
             mock.patch.object(self.n, '_start') as start:
            self.tick(); start.assert_not_called()
            ready.return_value = []
            self.tick(); start.assert_called_once_with(None)

    def test_readiness_timeout_cancels_no_automatic_retry(self):
        self.write(1, 'start', self.start_time-31)
        with mock.patch.object(self.n, '_ready', return_value=['head_rgb']), mock.patch.object(self.n, '_start') as start:
            self.tick(); self.tick()
            start.assert_not_called()
        self.assertTrue(self.n.session_done)
        self.assertIn('readiness_timeout', (self.folder/'session_result.json').read_text())

    def test_early_finish_before_subscriber_start_cancels(self):
        self.write(2, 'finish')
        with mock.patch.object(self.n, '_start') as start:
            self.tick(); start.assert_not_called()
        self.assertTrue(self.n.session_done)
        self.assertFalse(self.n.recorder.recording)

    def test_early_abort_cancels_without_starting(self):
        self.write(2, 'abort'); self.tick()
        self.assertTrue(self.n.session_done)

    def test_operator_discard_before_ready_cancels_without_start(self):
        self.write(2, 'discard')
        with mock.patch.object(self.n, '_start') as start:
            self.tick(); start.assert_not_called()
        self.assertEqual(json.loads((self.folder/'session_result.json').read_text())['outcome'], 'discarded')

    def test_operator_discard_preserves_empty_metadata_not_fault(self):
        self.fake_recording(); self.write(2, 'discard'); self.tick()
        path = next((self.n.source_root/'_discarded').iterdir())
        result = t.verify(path, allow_incomplete=True)
        self.assertEqual(result['readback'], 'METADATA_ONLY')
        self.assertTrue(result['discarded'])
        self.assertFalse((self.n.source_root/'_interrupted').exists())

    def test_operator_discard_preserves_rows_images_and_exclusion(self):
        self.fake_recording(); self.n.clock_offset = 90.
        f = t.fixture(); bundle, missing = t.select(f)
        self.assertFalse(missing)
        self.n._save_bundle(f[0], 10., bundle,
            t.JsonSample(9.9,99.9,{'current':{'name':self.n._required_state_name()}}))
        self.write(2, 'discard'); self.tick()
        path = next((self.n.source_root/'_discarded').iterdir())
        result = t.verify(path, allow_incomplete=True)
        self.assertEqual(result['decoded_images'], 3)
        self.assertTrue(result['discarded'])
        self.assertTrue(result['excluded_from_training'])
        self.assertFalse(result['episode_complete'])
        self.assertFalse((self.n.source_root/'episode_0000').exists())
        with self.assertRaises(ValueError): t.verify(path)

    def test_recording_finish_has_bounded_drain_not_instant_discard(self):
        self.fake_recording(); self.write(2, 'finish')
        self.tick()
        self.assertTrue(self.n.recorder.recording)
        self.assertLessEqual(self.n.finish_deadline, time.monotonic()+.4)
        self.assertEqual(self.n.finish_reason, 'unit_test')

    def test_very_short_clip_preserved_instead_of_refusing_to_stop(self):
        self.fake_recording(); self.write(2, 'finish'); self.tick()
        self.n.finish_deadline = time.monotonic()-1
        self.tick()
        self.assertFalse(self.n.recorder.recording)
        self.assertTrue((self.n.source_root/'_interrupted').exists())
        self.assertTrue(self.n.session_done)

    def test_exit_sonic_requests_normal_drain(self):
        self.fake_recording()
        self.n.latest_state_machine.value['current']['name'] = 'com.bxi.basic_actions/normal'
        self.tick()
        self.assertEqual(self.n.stop_reason, 'sonic_state_exit')
        self.assertLessEqual(self.n.finish_deadline, time.monotonic()+.4)

    def test_lost_state_stream_preserves_partial_and_does_not_resume(self):
        self.fake_recording()
        self.n.latest_state_machine = None
        self.tick()
        self.assertTrue(self.n.session_done)
        self.assertFalse(self.n.recorder.recording)
        self.assertTrue((self.n.source_root/'_interrupted').exists())

    def test_clock_failure_preserves_partial(self):
        self.fake_recording()
        with mock.patch.object(self.n, '_clock_ok', return_value=False): self.n._control_tick()
        self.assertTrue(self.n.session_done)

    def test_low_disk_stops_and_preserves(self):
        self.fake_recording()
        with mock.patch.object(self.n, '_clock_ok', return_value=True), \
             mock.patch.object(self.mod.shutil, 'disk_usage', return_value=SimpleNamespace(free=1024)):
            self.n._control_tick()
        self.assertTrue(self.n.session_done)
        self.assertIn('low_disk', (self.folder/'session_result.json').read_text())

    def test_duplicate_intent_is_not_repeated_and_future_rejected(self):
        with mock.patch.object(self.n, '_finish') as finish:
            self.write(2, 'finish'); self.n._consume_intent(self.start_time); self.n._consume_intent(self.start_time)
            finish.assert_called_once()
        self.write(3, 'start', self.start_time+3)
        with self.assertRaises(ValueError): self.n._consume_intent(self.start_time)

    def test_session_start_overrides_thirty_second_trial_limit(self):
        def started(*_): self.fake_recording()
        with mock.patch.object(self.mod.SourceCollectionNode, '_start', side_effect=started):
            self.n._start(None)
        self.assertAlmostEqual(self.n.capture_end_unix-self.n.recorder.started_unix, MAX_EPISODE_SECONDS)
        self.assertGreater(self.n.finish_deadline-time.monotonic(), 590)

    def test_public_ros_start_cannot_create_an_unrequested_episode(self):
        self.n.intent_sequence = 1
        self.n.manual_events.append(dict(command='start'))
        with mock.patch.object(self.n, '_start') as start:
            self.tick(); start.assert_not_called()

    def test_usb_silence_preserves_same_r3_fault_semantics(self):
        self.fake_recording()
        self.n._capture_tick()
        self.assertTrue(self.n.session_done)
        self.assertIn('camera_input_timeout', (self.folder/'session_result.json').read_text())

    def test_finalize_error_after_recording_flag_cleared_still_notifies_parent(self):
        self.fake_recording(); self.n.min_frames = 0
        self.n.finish_deadline = time.monotonic()-1
        def fail(*_): self.n.recorder.recording = False
        with mock.patch.object(self.mod.RealCollectionNode, '_finish', side_effect=fail):
            self.tick()
        self.assertTrue(self.n.session_done)
        self.assertIn('episode_finalization_failed', (self.folder/'session_result.json').read_text())
        # Restore recording for fixture cleanup of its synthetic writer thread.
        self.n.recorder.recording = True

    def test_real_writer_finish_saves_and_readback_decodes_every_image(self):
        self.fake_recording(); self.n.clock_offset = 90.
        for i in range(100):
            stamp = 100.+i/30.
            f = t.fixture(i, stamp); bundle, missing = t.select(f)
            self.assertFalse(missing)
            self.n._save_bundle(f[0], stamp-90., bundle,
                t.JsonSample(stamp-90.-.1,stamp-.1,{'current':{'name':self.n._required_state_name()}}))
            self.n.recorder.queue.join()
        self.n.latest_state_machine = t.JsonSample(time.monotonic(), time.time(),
            {'current':{'name':self.n._required_state_name()}})
        self.write(2, 'finish'); self.tick()
        self.n.finish_deadline = time.monotonic()-1
        self.tick()
        self.assertTrue(self.n.session_done)
        episode = self.n.source_root/'episode_0000'
        self.assertEqual(t.verify(episode)['decoded_images'], 300)
        self.assertEqual(json.loads((episode/'meta.json').read_text())['termination']['reason'], 'unit_test')

    def test_paused_initial_start_waits_without_readiness_timeout(self):
        self.write(2, 'pause', pauses=[[self.start_time-60, None]])
        with mock.patch.object(self.n, '_start') as start:
            self.tick()
            self.n.pending_start = self.start_time-1
            self.tick()
            start.assert_not_called()
        self.assertFalse(self.n.session_done)
        self.write(3, 'resume', pauses=[[self.start_time-60, self.start_time]])
        with mock.patch.object(self.n, '_ready', return_value=[]), mock.patch.object(self.n, '_start') as start:
            self.tick()
            start.assert_called_once_with(None)

    def test_pause_budget_excludes_only_pause_after_recording_start(self):
        self.fake_recording()
        self.n.recording_started_mono = 10.
        self.n.pauses = [[5., 12.], [20., None]]
        original = self.n.finish_deadline
        self.n._update_pause_budget(720.)
        self.assertEqual(self.n.finish_deadline-original, 702.)
        self.n._update_pause_budget(720.)
        self.assertEqual(self.n.finish_deadline-original, 702.)
        self.n.pauses[-1][1] = 725.
        self.n._update_pause_budget(730.)
        self.assertEqual(self.n.finish_deadline-original, 707.)

    def test_pause_does_not_finalize_on_old_wall_deadline(self):
        self.fake_recording()
        self.write(2, 'pause', pauses=[[self.start_time-10, None]])
        self.n.finish_deadline = self.start_time-1
        self.tick()
        self.assertTrue(self.n.recorder.recording)
        self.assertFalse(self.n.session_done)
        self.write(3, 'discard'); self.tick()
        self.assertTrue(self.n.session_done)
        self.assertTrue((self.n.source_root/'_discarded').exists())

    def test_camera_fault_while_paused_still_preserves_interrupted(self):
        self.fake_recording()
        self.write(2, 'pause', pauses=[[self.start_time, None]])
        self.n._capture_tick()
        self.assertTrue(self.n.session_done)
        self.assertIn('camera_input_timeout', (self.folder/'session_result.json').read_text())

    def test_fast_pause_resume_keeps_every_boundary_and_rejects_rewrite(self):
        history = [[self.start_time-2, self.start_time-1], [self.start_time-.5, self.start_time]]
        self.write(5, 'resume', pauses=history)
        self.n._consume_intent(self.start_time)
        self.assertEqual(self.n.pauses, history)
        self.assertFalse(self.n.paused)
        self.write(6, 'resume', pauses=[])
        with self.assertRaisesRegex(ValueError, 'rewritten'):
            self.n._consume_intent(self.start_time)

    def test_paused_and_cross_boundary_bundles_write_nothing(self):
        self.fake_recording(); self.n.clock_offset = 90.
        self.n.pauses = [[12., 17.]]
        for stamp in (104., 107.005):
            f = t.fixture(1, stamp); bundle, missing = t.select(f)
            self.assertFalse(missing)
            self.assertFalse(self.n._save_bundle(f[0], stamp-90., bundle, None))
        self.assertEqual(self.n.recorder.frame_count, 0)
        self.assertFalse(self.n._anchor_allowed(t.fixture(1, 104.)[0]))
        self.assertTrue(self.n._anchor_allowed(t.fixture(1, 101.)[0]))  # drain queued pre-pause rows

    def test_capture_timer_drains_pre_pause_skips_pause_and_resumes(self):
        from collections import deque
        self.fake_recording(); self.n.clock_offset = 90.
        self.n.source_buffer.reset(100.)
        self.write(2, 'pause', when=12., pauses=[[12., None]])
        for i, (stamp, expected_rows) in enumerate(((101.9,1), (104.,1), (107.005,1), (107.1,2))):
            if i == 2:
                self.write(3, 'resume', when=17., pauses=[[12., 17.]])
            anchor, histories, images, pico, smpl = t.fixture(i, stamp)
            for key, attr in (('state','states'), ('command','commands'), ('imu','imus'),
                              ('gripper','gripper_states'), ('inference','sonic_inferences')):
                setattr(self.n, attr, deque(histories[key]))
            self.n.images = {key:deque(value) for key,value in images.items()}
            self.n.status_history = deque([t.JsonSample(stamp-90.-.1, stamp-.1,
                {'current':{'name':self.n._required_state_name()}})])
            self.n.pico.after.return_value = (pico,)
            self.n.smpl.after.return_value = (smpl,)
            with mock.patch.object(self.mod.time, 'monotonic', return_value=stamp-90.+.3), \
                 mock.patch.object(self.mod.time, 'time', return_value=stamp+.3), \
                 mock.patch.object(self.n, '_clock_ok', return_value=True):
                self.n._capture_tick()
            self.n.recorder.queue.join()
            self.assertFalse(self.n.session_done)
            self.assertEqual(self.n.recorder.frame_count, expected_rows)

    def test_save_one_episode_across_pause_with_real_timestamp_gap(self):
        self.fake_recording(); self.n.clock_offset = 90.
        self.write(3, 'resume', pauses=[[12., 17.]])
        self.n._consume_intent(self.start_time)
        for i in range(100):
            stamp = 100.+i/30. if i < 50 else 107.1+(i-50)/30.
            f = t.fixture(i, stamp); bundle, missing = t.select(f)
            self.assertFalse(missing)
            self.assertTrue(self.n._save_bundle(f[0], stamp-90., bundle,
                t.JsonSample(stamp-90.-.1, stamp-.1, {'current':{'name':self.n._required_state_name()}})))
            self.n.recorder.queue.join()
        # Can explicitly save even during a later pause, without creating another episode.
        self.n.latest_state_machine = t.JsonSample(time.monotonic(), time.time(),
            {'current':{'name':self.n._required_state_name()}})
        self.write(4, 'pause', pauses=[[12., 17.], [19., None]])
        self.tick()
        self.write(5, 'finish'); self.tick()
        self.n.finish_deadline = time.monotonic()-1
        self.n.latest_state_machine = t.JsonSample(time.monotonic(), time.time(),
            {'current':{'name':self.n._required_state_name()}})
        self.tick()
        episode = self.n.source_root/'episode_0000'
        report = t.verify(episode)
        self.assertEqual(report['decoded_images'], 300)
        self.assertEqual(report['saved_rows'], 100)
        self.assertEqual(report['recorded_segments'], 2)
        self.assertEqual(report['segment_boundaries'], [50])
        self.assertEqual(report['timing_review'], 'PASS')
        self.assertGreater(report['maximum_row_gap_s'], 5.)
        self.assertAlmostEqual(report['uninterrupted_row_rate_hz'], 30.)
        self.assertEqual(len(list(self.n.source_root.glob('episode_*'))), 1)
        rows = [json.loads(line) for line in (episode/'frames.jsonl').read_text().splitlines()]
        self.assertGreater(rows[50]['t_episode']-rows[49]['t_episode'], 5.)
        # The boundary marker is enforced by the independent readback.
        rows[50]['auxiliary']['capture_segment'] = 0
        (episode/'frames.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
        with self.assertRaisesRegex(ValueError, 'wrong segment'):
            t.verify(episode)


if __name__ == '__main__': unittest.main()
