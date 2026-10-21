"""Real PyAV flush/error tests and upstream reuse checks, without any robot I/O."""
import hashlib
import inspect
import json
from pathlib import Path
import tempfile
import threading
import unittest

import av
import numpy as np

from vendor.sonic_data.video_writer import VideoWriter
import official_writer


class OfficialWriterTests(unittest.TestCase):
    def test_gr00t_statistics_function_is_unchanged(self):
        from vendor.gr00t_stats import calculate_dataset_statistics
        source = inspect.getsource(calculate_dataset_statistics).rstrip("\n")
        meta = json.loads((Path(__file__).parent / "vendor/GR00T_STATS_UPSTREAM.json").read_text())
        self.assertEqual(hashlib.sha256(source.encode()).hexdigest(), meta["function_sha256"])

    def test_upstream_exporter_only_import_path_changed(self):
        folder = Path(__file__).parent / "vendor/sonic_data"
        source = (folder / "exporter.py").read_text().replace(
            "from .video_writer import VideoWriter",
            "from gear_sonic.data.video_writer import VideoWriter")
        manifest = json.loads((folder / "UPSTREAM.json").read_text())
        self.assertEqual(hashlib.sha256(source.encode()).hexdigest(),
                         manifest["files"]["gear_sonic/data/exporter.py"]["upstream_sha256"])

    def test_waits_for_inflight_last_frame_not_only_empty_queue(self):
        with tempfile.TemporaryDirectory() as tmp:
            writer = VideoWriter(Path(tmp) / "tail.mp4", 32, 32, 50)
            encoding = threading.Event()
            release = threading.Event()
            done = threading.Event()
            original = writer._flush_stream
            def blocked_flush():
                encoding.set()
                if not release.wait(5):
                    raise RuntimeError("test timed out")
                original()
            writer._flush_stream = blocked_flush
            for i in range(7):
                writer.add_frame(np.full((32, 32, 3), i * 30, np.uint8))
            errors = []
            def stop():
                try:
                    writer.stop()
                except Exception as exc:
                    errors.append(exc)
                finally:
                    done.set()
            thread = threading.Thread(target=stop)
            thread.start()
            try:
                self.assertTrue(encoding.wait(5))
                self.assertFalse(done.wait(.05))
            finally:
                release.set()
                thread.join(5)
            self.assertTrue(done.is_set())
            self.assertFalse(errors)
            self.assertFalse(writer._thread.is_alive())
            writer.stop()  # Closing twice is safe.
            with av.open(str(writer.output_path)) as container:
                frames = list(container.decode(video=0))
            self.assertEqual(len(frames), 7)
            self.assertGreater(frames[-1].to_ndarray(format="rgb24").mean(), 170)

    def test_encoder_failure_is_reported_and_terminates(self):
        with tempfile.TemporaryDirectory() as tmp:
            writer = VideoWriter(Path(tmp) / "error.mp4", 32, 32, 50)
            def fail():
                raise OSError("synthetic encoder flush failure")
            writer._flush_stream = fail
            writer.add_frame(np.zeros((32, 32, 3), np.uint8))
            with self.assertRaisesRegex(RuntimeError, "Video encoder failed"):
                writer.stop()
            self.assertFalse(writer._thread.is_alive())

    def test_full_queue_waits_then_saves_every_frame(self):
        from unittest.mock import patch
        import queue
        import time
        with tempfile.TemporaryDirectory() as tmp:
            writer = VideoWriter(Path(tmp)/"busy.mp4", 32, 32, 50, buffer_size=2)
            entered, release, complete = threading.Event(), threading.Event(), threading.Event()
            original = writer._writer_worker
            def delayed_worker():
                entered.set()
                release.wait(5)
                original()
            writer._writer_worker = delayed_worker
            errors = []
            def produce():
                try:
                    for i in range(10): writer.add_frame(np.full((32,32,3),i*20,np.uint8))
                except Exception as exc: errors.append(exc)
                finally: complete.set()
            producer = threading.Thread(target=produce)
            producer.start()
            try:
                self.assertTrue(entered.wait(3))
                self.assertFalse(complete.wait(.2))
                self.assertEqual(writer.queue.qsize(), 2)
            finally:
                release.set()
                producer.join(5)
            self.assertTrue(complete.is_set())
            self.assertFalse(errors)
            writer.stop()
            with av.open(str(writer.output_path)) as stream:
                self.assertEqual(sum(1 for _ in stream.decode(video=0)), 10)

    def test_idle_does_not_create_file_or_worker(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "empty.mp4"
            writer = VideoWriter(path, 32, 32, 50)
            writer.stop()
            self.assertFalse(path.exists())
            self.assertIsNone(writer._thread)

    def test_official_discard_flag_and_rgb(self):
        from contract import CAMERAS, VECTOR_NAMES
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "dataset"
            writer = official_writer.create(root, "SYNTHETIC TEST: move red object")
            try:
                frame = {k: np.zeros(len(names), np.float32) for k, names in VECTOR_NAMES.items()}
                rgb = np.zeros((480, 640, 3), np.uint8)
                rgb[:, :, 0] = 255
                for camera in CAMERAS:
                    frame["observation.images." + camera] = rgb
                for _ in range(5):
                    writer.add_frame(frame)
                writer.save_episode_as_discarded()
                self.assertEqual(official_writer.verify_videos(writer), 15)
            finally:
                official_writer.close(writer)
            self.assertEqual(json.loads((root / "meta/info.json").read_text())["discarded_episode_indices"], [0])
            path = next(root.rglob("*.mp4"))
            with av.open(str(path)) as container:
                rgb = next(container.decode(video=0)).to_ndarray(format="rgb24")
            self.assertGreater(rgb[:, :, 0].mean(), 240)
            self.assertLess(rgb[:, :, 2].mean(), 10)


if __name__ == "__main__":
    unittest.main()
