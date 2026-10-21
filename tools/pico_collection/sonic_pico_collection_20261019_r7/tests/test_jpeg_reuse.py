"""JPEG reuse must preserve pixels, source boundaries and episode isolation."""
from pathlib import Path
from queue import Queue
import tempfile
import unittest
from unittest import mock

import test_trial  # Set package paths without ROS initialization.
import cv2
import numpy as np
from configure import COLLECTOR_RELEASE, configuration
from real_robot_collection.recorder import RealEpisodeRecorder


def writer(root, records):
    node = object.__new__(RealEpisodeRecorder)
    node.common_metadata = {"capture_profile": "sonic50_latest_rgb"}
    node.queue = Queue()
    node.stop_worker, node.discarding = True, False
    node.inprogress_dir, node.jpeg_quality = root, 90
    node.rows, node.errors = [], []
    for camera in ("head_rgb", "left_wrist_rgb", "right_wrist_rgb"):
        (root / camera / "rgb").mkdir(parents=True)
    for index, (segment, sequence, stamp, pixel) in enumerate(records):
        cameras = ("head_rgb", "left_wrist_rgb", "right_wrist_rgb")
        node.queue.put(dict(row=dict(idx=index,
            auxiliary=dict(capture_segment=segment,
                           selected_camera_sequences={n:sequence for n in cameras}),
            source_timestamps_unix={n:stamp for n in cameras}),
            rgb={n:np.full((16, 16, 3), pixel+j, np.uint8) for j,n in enumerate(cameras)},
            depth=None))
    return node


class JpegReuseTests(unittest.TestCase):
    def test_cache_keys_preserve_pixels_across_sources_pauses_and_episodes(self):
        records = [(0,1,10.,20), (0,1,10.,20), (0,2,11.,60),
                   (1,2,11.,100), (1,2,11.,100), (1,2,12.,150)]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            node = writer(root/"first", records)
            with mock.patch.object(cv2, "imencode", wraps=cv2.imencode) as encode:
                node._worker()
            self.assertFalse(node.errors)
            self.assertEqual(len(node.rows), 6)
            self.assertEqual(encode.call_count, 4*3)
            for i, record in enumerate(records):
                for j,camera in enumerate(("head_rgb","left_wrist_rgb","right_wrist_rgb")):
                    image = cv2.imread(str(root/"first"/camera/"rgb"/f"{i:06d}.jpg"))
                    self.assertTrue(np.all(image == record[3]+j))
            # Same source ID in another episode still encodes new pixels.
            other = writer(root/"second", [(0,1,10.,200)])
            with mock.patch.object(cv2, "imencode", wraps=cv2.imencode) as encode:
                other._worker()
            self.assertEqual(encode.call_count, 3)
            self.assertTrue(np.all(cv2.imread(str(root/"second/head_rgb/rgb/000000.jpg")) == 200))

    def test_encode_failure_is_recorded_and_not_published_as_a_saved_row(self):
        with tempfile.TemporaryDirectory() as directory:
            node = writer(Path(directory), [(0,1,10.,20)])
            with mock.patch.object(cv2, "imencode", return_value=(False, None)):
                node._worker()
            self.assertFalse(node.rows)
            self.assertIn("failed to encode", node.errors[0])
            self.assertEqual(node.queue.unfinished_tasks, 0)

    def test_status_uses_code_release_not_legacy_directory_name(self):
        from button_supervisor import Supervisor
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"status.json"
            node = Supervisor(directory, path)
            node.publish_status()
            import json
            self.assertEqual(json.loads(path.read_text())["collector_release"], COLLECTOR_RELEASE)
            self.assertEqual(configuration()["metadata"]["collector_release"], COLLECTOR_RELEASE)
