"""50 Hz numeric rows, 30 Hz RGB, real writer, pause gap and final readback."""
import importlib
import json
from pathlib import Path
import tempfile
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import yaml
import test_latest_rgb as rgb

class Paused50Hz(unittest.TestCase):
    def test_saved_and_discarded_both_keep_one_episode_with_two_segments(self):
        base = rgb.t.load_node()
        with patch.dict(sys.modules, {'real_robot_collection.node':SimpleNamespace(RealCollectionNode=base.RealCollectionNode),
                                     'real_robot_collection.source_node':base}):
            module = importlib.import_module('session_node')
        for discard in (False, True):
            with self.subTest(discard=discard), tempfile.TemporaryDirectory() as temp:
                cfg = rgb.config50(); cfg['dataset']['output_root'] = temp
                path = Path(temp)/'cfg.yaml'; path.write_text(yaml.safe_dump(cfg))
                with patch.dict(base.RealCollectionNode.__init__.__globals__,
                                PicoSubscriber=rgb.mock.Mock(),smpl_subscriber=rgb.mock.Mock()):
                    n = module.SessionNode(path,'SYNTHETIC_PAUSE_ONLY',temp)
                try:
                    n.recorder.start_episode(dict(task_id='TEST',instruction_text='SYNTHETIC ONLY'),runtime_metadata={})
                    n.clock_offset = 90.; n.pauses = [[11.2,16.2]]
                    for i in range(120):
                        f = rgb.source50(i if i < 60 else i+255)
                        a = f[0]; bundle,missing = rgb.choose(f)
                        self.assertFalse(missing)
                        status = rgb.t.JsonSample(a.stamp_unix-90-.01,a.stamp_unix-.01,
                                                  {'current':{'name':n._required_state_name()}})
                        self.assertTrue(n._save_bundle(a,a.stamp_unix-90,bundle,status))
                        n.last_selected_camera_sequences = {name:image.sequence for name,image in bundle['rgb'].items()}
                        n.recorder.queue.join()
                    n.recorder.meta['synchronization'] = n.sync_summary()
                    if discard:
                        n._discard('synthetic_operator_discard')
                        episode = next((n.source_root/'_discarded').iterdir())
                    else:
                        episode = Path(n.recorder.end_episode({'success':True},100)['path'])
                    report = rgb.verify(episode,allow_incomplete=discard)
                    self.assertEqual(report['saved_rows'],120)
                    self.assertEqual(report['decoded_images'],360)
                    self.assertEqual(report['recorded_segments'],2)
                    self.assertEqual(report['segment_boundaries'],[60])
                    self.assertAlmostEqual(report['uninterrupted_row_rate_hz'],50)
                    self.assertEqual(report['timing_review'],'INCOMPLETE' if discard else 'PASS')
                    if discard: self.assertTrue(report['excluded_from_training'])
                finally:
                    n.close()
