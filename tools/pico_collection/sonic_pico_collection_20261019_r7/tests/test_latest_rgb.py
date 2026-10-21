"""Real recorder code, synthetic transport/images only. No ROS init or device access."""
import copy
from dataclasses import replace
import importlib
import json
import os
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

import numpy as np
import yaml

import test_trial as t
from configure import configuration
from real_robot_collection.source_sync import AnchorBuffer, choose_bundle
from real_robot_collection.gripper_identity import identities
from verify_saved import verify


def config50():
    site = json.loads((t.ROOT.parents[2] / "tools/customer/site.example.json").read_text())
    site.update(capture_profile="sonic50_latest_rgb", camera_roles_confirmed=True,
                camera_role_evidence="SYNTHETIC TEST ONLY",
                training_task="SYNTHETIC TEST ONLY: 把桌上的盒子放到右侧托盘")
    with mock.patch("configure.site", return_value=site):
        cfg = configuration("bxi", "bxi")
        cfg["metadata"]["storage_backend"] = "legacy_jpeg"  # Historical raw-JPEG regression fixture.
        return cfg


def source50(i=0):
    stamp = 100 + i / 50
    a, h, images, p, s = t.fixture(i, stamp)
    h["inference"] = [replace(h["inference"][0], stamp_unix=stamp)]
    h["command"] = [replace(h["command"][0], stamp_unix=stamp + .001)]
    raw = h["gripper"][0].raw
    for item, identity in zip(raw["samples"], identities("bxi", "bxi")):
        item.update(identity, open_rad=float(np.deg2rad(300)),
                    closure_fraction=1-2.5/np.deg2rad(300),
                    command_closure_fraction=1-1/np.deg2rad(300))
    ci = int(np.floor(i * 30 / 50 + 1e-7))
    for name, values in images.items():
        values[0].data.fill(ci % 255)
        images[name] = [replace(values[0], sequence=ci, stamp_unix=100 + ci / 30)]
    buf = AnchorBuffer(wait_s=.25)
    anchor = buf.ready_inference(h["inference"], stamp + .26)
    return anchor, h, images, p, s


def choose(f, cameras=None, last=-1):
    a, h, images, p, s = f
    return choose_bundle(a, h, images, cameras or {n:-1 for n in t.WRISTS},
                         last, [p], [s], latest_rgb=True)


class LatestRGBTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = t.load_node()

    def test_config_50hz_keeps_physical_cameras_30hz_and_task(self):
        cfg = config50()
        t.validate_config(cfg, formal=False, allow_placeholders=False)
        self.assertEqual(cfg["dataset"]["fps"], 50)
        for n in t.WRISTS:
            self.assertEqual(cfg["cameras"][n]["fps"], 30)
        self.assertIn("盒子", cfg["dataset"]["tasks"][0]["instruction"])

    def test_latest_image_is_reused_but_inference_is_not(self):
        first, _ = choose(source50(0))
        previous = {n:x.sequence for n,x in first["rgb"].items()}
        second, missing = choose(source50(1), previous, last=0)
        self.assertFalse(missing)
        self.assertEqual(second["inference"].inference_sequence, 1)
        self.assertEqual(second["rgb"]["head_rgb"].sequence, 0)
        self.assertIn("matched_sonic_inference", choose(source50(1), previous, last=1)[1])

    def test_no_future_stale_or_backward_images(self):
        for stamp in (100.1, 99.89):
            f = source50()
            n = "head_rgb"
            f[2][n] = [replace(f[2][n][0], stamp_unix=stamp)]
            self.assertIn(n, choose(f)[1])
        self.assertIn("head_rgb", choose(source50(), {n:2 for n in t.WRISTS})[1])

    def test_no_held_reference_or_mismatched_command(self):
        f = source50()
        buf = AnchorBuffer()
        self.assertIsNone(buf.ready_inference(
            [replace(f[1]["inference"][0], policy_status="stale_hold")], 100.3))
        f[1]["command"] = [replace(f[1]["command"][0], position=np.ones(31))]
        self.assertIn("command", choose(f)[1])

    def test_pose_resume_rejects_pre_pause_camera(self):
        with mock.patch.dict(sys.modules, {
                "real_robot_collection.node": SimpleNamespace(RealCollectionNode=self.mod.RealCollectionNode),
                "real_robot_collection.source_node": self.mod}):
            module = importlib.import_module("session_node")
        n = object.__new__(module.SessionNode)
        n.clock_offset = 90.
        n.pauses = [[10.005, 10.015]]
        f = source50(1)
        bundle, missing = choose(f)
        self.assertFalse(missing)
        # Camera at 100.0 predates pause; inference at 100.02 follows resume.
        self.assertFalse(n._save_bundle(f[0], 10.02, bundle, None))

    def test_real_writer_50_rows_30_images_readback_and_tamper(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg = config50()
            cfg["dataset"]["output_root"] = temp
            cfgpath = Path(temp) / "trial.yaml"
            cfgpath.write_text(yaml.safe_dump(cfg, allow_unicode=True))
            with mock.patch("real_robot_collection.node.PicoSubscriber"), mock.patch(
                    "real_robot_collection.node.smpl_subscriber"):
                n = self.mod.SourceCollectionNode(cfgpath, "SYNTHETIC_50HZ_TEST")
            try:
                task = cfg["dataset"]["tasks"][0]["instruction"]
                n.recorder.start_episode(dict(task_id="TEST", instruction_text=task), runtime_metadata={})
                n.source_buffer.reset(100.)
                n.clock_offset = 90.
                for i in range(120):
                    a, h, images, p, s = source50(i)
                    for attr, key in (("states", "state"), ("commands", "command"), ("imus", "imu"),
                                      ("gripper_states", "gripper"), ("sonic_inferences", "inference")):
                        getattr(n, attr).extend(h[key])
                    for name, values in images.items():
                        n.images[name].extend(values)
                    n.pico.after.return_value = [p]
                    n.smpl.after.return_value = [s]
                    n.status_history.append(t.JsonSample(a.stamp_unix-90-.01, a.stamp_unix-.01,
                                                        {"current":{"name":n._required_state_name()}}))
                    with mock.patch.object(self.mod.time, "time", return_value=a.stamp_unix+.26), \
                         mock.patch.object(self.mod.time, "monotonic", return_value=a.stamp_unix-90+.26):
                        n._capture_tick()
                    n.recorder.queue.join()
                self.assertEqual(n.recorder.frame_count, 120)
                n.recorder.meta["synchronization"] = n.sync_summary()
                path = Path(n.recorder.end_episode({"success": True}, 100)["path"])
                result = verify(path)
                self.assertEqual(result["timing_review"], "PASS")
                self.assertAlmostEqual(result["uninterrupted_row_rate_hz"], 50)
                self.assertEqual(result["camera_reused_rows"], {k:48 for k in t.WRISTS})
                meta = json.loads((path / "meta.json").read_text())
                self.assertEqual(meta["task_name"], task)
                self.assertEqual(meta["timebase"]["row_clock"], "sonic_inference_source_stamp")
                self.assertEqual(meta["gripper_coordinate"], "absolute_motor_radians")
                rows = [json.loads(x) for x in (path / "frames.jsonl").read_text().splitlines()]
                self.assertEqual(len({r["sonic.inference"]["sequence"] for r in rows}), 120)
                output = os.environ.get("SYNTHETIC_50HZ_OUTPUT")
                if output:
                    shutil.copytree(path, output)  # Persistent input for the separate GR00T environment.
                rows[1]["auxiliary"]["camera_reused"]["head_rgb"] = False
                (path / "frames.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))
                with self.assertRaisesRegex(ValueError, "reuse audit"):
                    verify(path)
            finally:
                n.close()


if __name__ == "__main__":
    unittest.main()
