"""50 Hz numeric rows, 30 Hz RGB, real writer, pause gap and final readback."""
import importlib
import itertools
import json
from pathlib import Path
import tempfile
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import yaml
import test_latest_rgb as rgb

class OfficialOnline(unittest.TestCase):
    def test_saved_and_discarded_both_keep_one_episode_with_two_segments(self):
        base = rgb.t.load_node()
        with patch.dict(sys.modules, {'real_robot_collection.node':SimpleNamespace(RealCollectionNode=base.RealCollectionNode),
                                     'real_robot_collection.source_node':base}):
            module = importlib.import_module('session_node')
        for discard, paused in itertools.product((False, True), repeat=2):
            with self.subTest(discard=discard, paused=paused), tempfile.TemporaryDirectory() as temp:
                cfg = rgb.config50(); cfg['metadata']['storage_backend'] = 'nvidia_sonic_lerobot_v1'; cfg['dataset']['output_root'] = temp
                path = Path(temp)/'cfg.yaml'; path.write_text(yaml.safe_dump(cfg))
                with patch.dict(base.RealCollectionNode.__init__.__globals__,
                                PicoSubscriber=rgb.mock.Mock(),smpl_subscriber=rgb.mock.Mock()):
                    n = module.SessionNode(path,'SYNTHETIC_PAUSE_ONLY',temp)
                try:
                    n.recorder.start_episode(dict(task_id='TEST',instruction_text='SYNTHETIC ONLY'),runtime_metadata={})
                    self.assertIsInstance(n.recorder, __import__('real_robot_collection.official_recorder', fromlist=['OfficialEpisodeRecorder']).OfficialEpisodeRecorder)
                    n.clock_offset = 90.; n.pauses = [[11.2,16.2]] if paused else []
                    for i in range(120):
                        f = rgb.source50(i+255 if paused and i >= 60 else i)
                        a = f[0]; bundle,missing = rgb.choose(f)
                        self.assertFalse(missing)
                        status = rgb.t.JsonSample(a.stamp_unix-90-.01,a.stamp_unix-.01,
                                                  {'current':{'name':n._required_state_name()}})
                        self.assertTrue(n._save_bundle(a,a.stamp_unix-90,bundle,status))
                        n.last_selected_camera_sequences = {name:image.sequence for name,image in bundle['rgb'].items()}
                        self.assertIsNone(n.recorder.queue)
                    n.recorder.meta['synchronization'] = n.sync_summary()
                    if discard:
                        n._discard('synthetic_operator_discard')
                        episode = next((n.source_root/'_discarded').iterdir())
                    else:
                        episode = Path(n.recorder.end_episode({'success':True},100)['path'])
                    report = rgb.verify(episode,allow_incomplete=discard)
                    self.assertEqual(report['saved_rows'],120)
                    info = json.loads((episode/'lerobot/meta/info.json').read_text())
                    self.assertEqual(info['total_episodes'], 1)
                    self.assertEqual(info['discarded_episode_indices'], [0] if discard else [])
                    self.assertFalse(list(episode.rglob('*.jpg')))
                    self.assertEqual(len(list(episode.rglob('*.mp4'))), 3)
                    self.assertEqual(report['decoded_images'],360)
                    self.assertEqual(report['recorded_segments'],2 if paused else 1)
                    self.assertEqual(report['segment_boundaries'],[60] if paused else [])
                    self.assertAlmostEqual(report['uninterrupted_row_rate_hz'],50)
                    self.assertEqual(report['timing_review'],'INCOMPLETE' if discard else 'PASS')
                    if discard: self.assertTrue(report['excluded_from_training'])
                    # Detect a Parquet/provenance disagreement, not only valid MP4 files.
                    sidecar = episode/'frames.jsonl'
                    original = sidecar.read_text()
                    lines = original.splitlines()
                    row = json.loads(lines[0]); row['action.motion_token'][0] += .125
                    lines[0] = json.dumps(row); sidecar.write_text('\n'.join(lines)+'\n')
                    try:
                        with self.assertRaisesRegex(ValueError, 'Parquet differs'):
                            rgb.verify(episode, allow_incomplete=discard)
                    finally: sidecar.write_text(original)
                finally:
                    n.close()
