"""Synthetic saved-episode tests: no ROS, cameras, CAN, or training jobs."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np
import pyarrow.parquet as pq

from contract import CAMERAS, FPS, JOINTS, RAW_JOINTS, MODEL_SHA, VECTOR_NAMES
from legacy_export import episode_sequences, export, inspect_source, modality, training_image


def fixture(folder, paused=True):
    folder.mkdir()
    cameras = list(CAMERAS.values())
    identities = [
        dict(side=s, bus=b, motor_id=1, protocol="bxi", profile="bxi_colleague_rad_v1",
             position_unit="rad", effort_unit="Nm")
        for s, b in (("left", 5), ("right", 6))
    ]
    rows = []
    for i in range(120):
        t = 100 + i / FPS + (5 if paused and i >= 60 else 0)
        mono = t - 90
        native = dict(schema=1, monotonic=mono - .005, training_eligible=True,
                      ineligible_reasons=[], samples=[])
        for identity in identities:
            native["samples"].append(dict(
                **identity, open_rad=float(np.deg2rad(300)), closed_rad=0,
                position_rad=2.5, feedback_monotonic=mono - .01,
                closure_fraction=1 - 2.5 / np.deg2rad(300),
                command_closure_fraction=1 - 1 / np.deg2rad(300),
                command=dict(position=1, sent_monotonic=mono - .01),
            ))
        images = {}
        for camera in cameras:
            relative = f"{camera}/rgb/{i:06}.jpg"
            path = folder / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(path), np.full((16, 16, 3), i, dtype=np.uint8))
            images[camera] = relative
        sources = {k: t - .01 for k in (
            "actuator_state", "actuator_command", "imu", "sonic_inference",
            "gripper_state", "gripper_command", "pico", "smpl_ref")}
        sources["gripper_state"] = sources["gripper_command"] = t - .005
        sources.update({k: t for k in cameras})
        sources.update({s + "_gripper_" + k: t - .01
                        for s in ("left", "right") for k in ("feedback", "command")})
        state = [0.] * 29 + [.1, .2]
        action = [0.] * 29 + [.12, .22]
        row = {
            "idx": i, "diagnostic_only": True, "t_episode": t - 100,
            "dataset.observation.state": None, "dataset.action.wbc": None,
            "observation.state": state, "action": action, "action.motion_token": [0.] * 64,
            "observation.joint_present": [True] * 31, "action.joint_present": [True] * 31,
            "observation.projected_gravity": [0., 0., -1.],
            "observation.imu": dict(orientation_wxyz=[1, 0, 0, 0],
                                   angular_velocity=[0, 0, 0], linear_acceleration=[0, 0, 0]),
            "sonic.inference": dict(sequence=i, policy_status="live_reference",
                                    model_sha256=MODEL_SHA, target_position=action),
            "gripper": dict(
                coordinate="motor_rad_diagnostic_only", mapping={"verified": False},
                state=[2.5, 2.5], command=[1., 1.], native_raw_sample=native,
                state_is_measured=True, command_is_final=True,
            ),
            "observation.images": images, "source_timestamps_unix": sources,
            "auxiliary": dict(
                anchor_stamp_unix=t, capture_segment=int(paused and i >= 60),
                selected_camera_sequences={k: i for k in cameras},
                received_monotonic={k: mono + .02 for k in cameras}, clock_offset_s=90,
            ),
            "pico.raw": dict(frame_index=[i], timestamp_realtime=[t - .01]),
            "pico.smpl_ref": dict(frame_index=[i]),
        }
        rows.append(row)
    meta = dict(
        schema_version="elf3_three_rgb_gripper_source/v2", training_export_allowed=False,
        fps=FPS, termination={"success": True}, joint_contract=dict(canonical_names=RAW_JOINTS),
        metadata=dict(camera_left_right_verified=True, motion_token={"sha256": MODEL_SHA},
                      gripper_identities=identities, physical_ports={"left": 7, "right": 1}),
        gripper_identity_contract=identities, synchronization=dict(
            clock_jump=False, pose_pauses=[[101.2, 106.19]] if paused else []),
        camera_contract=dict(mounts={k: dict(height=16, width=16) for k in cameras}),
        num_frames=120, duration_s=rows[-1]["t_episode"],
        task_name="SYNTHETIC SOFTWARE TEST ONLY: move the object",
    )
    (folder / "meta.json").write_text(json.dumps(meta))
    write_rows(folder, rows)
    return meta, rows


def write_rows(folder, rows):
    (folder / "frames.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.raw = self.root / "episode_0000"
        self.meta, self.rows = fixture(self.raw)

    def tearDown(self):
        self.tmp.cleanup()

    def test_actual_export_pause_and_field_contract(self):
        out = self.root / "training"
        result = export([self.raw], out)
        self.assertEqual(result["training_sequences"], 1)
        self.assertEqual(result["frames"], 120)
        frame = pq.read_table(out / "data/chunk-000/episode_000000.parquet")
        self.assertEqual(set(frame.column_names),
                         set(VECTOR_NAMES) | {"index", "frame_index", "task_index",
                                             "episode_index", "timestamp"})
        self.assertEqual(frame["observation.state"].type.list_size, 31)
        self.assertNotIn("action.head_position", frame.column_names)
        self.assertEqual(frame["action.wbc"].type.list_size, 31)
        np.testing.assert_allclose(frame["action.wbc"][0].as_py()[-2:],
                                   [1.] * 2)
        np.testing.assert_allclose(frame["observation.state"][0].as_py()[-2:],
                                   [2.5] * 2)
        contract = json.loads((out / "meta/elf3_contract.json").read_text())
        self.assertEqual(contract["gripper_coordinate"], "absolute_motor_radians")
        self.assertTrue((out / "meta/native_1d_derivation.json").is_file())
        stats = [json.loads(line) for line in (out / "meta/episodes_stats.jsonl").read_text().splitlines()]
        self.assertEqual(len(stats), 1)
        self.assertEqual(stats[0]["stats"]["observation.state"]["count"], [120])
        self.assertEqual(stats[0]["stats"]["observation.state"]["mean"][-2:], [2.5, 2.5])
        # Upstream SONIC deliberately computes only non-video statistics.
        self.assertTrue(all(not k.startswith("observation.images.") for k in stats[0]["stats"]))
        self.assertEqual(result["storage_backend"], "NVIDIA_SONIC_Gr00tDataExporter")
        self.assertEqual(result["decoded_video_frames"], 360)
        self.assertEqual(len(list(out.rglob("*.mp4"))), 3)  # No empty next-episode videos.
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
        dataset = LeRobotDataset("tmp/tmp_dataset", root=out, video_backend="pyav")
        self.assertEqual(len(dataset), 120)
        for i in (0, 59, 60, 119):
            sample = dataset[i]
            self.assertEqual(sample["observation.state"].shape, (31,))
            for camera in CAMERAS:
                self.assertEqual(sample["observation.images." + camera].shape, (3, 480, 640))
        source = json.loads((out / "meta/source_map.json").read_text())
        self.assertEqual(len(source["sequences"]), 1)
        sequence = source["sequences"][0]
        self.assertEqual(sequence["capture_segments"], [0] * 60 + [1] * 60)
        self.assertEqual(sequence["source_row_indices"], list(range(120)))
        self.assertGreater(sequence["source_unix_s"][60] - sequence["source_unix_s"][59], 5)
        np.testing.assert_allclose(frame["timestamp"].to_numpy(), np.arange(120) / FPS, atol=1e-6)
        self.assertEqual(frame["episode_index"].to_pylist(), [0] * 120)
        self.assertEqual(frame["frame_index"].to_pylist(), list(range(120)))
        self.assertEqual(len(source["sources"]), 1)
        m = modality()
        self.assertEqual(m["annotation"]["human.task_description"]["original_key"], "task_index")
        with self.assertRaises(FileExistsError):
            export([self.raw], out)

    def test_inspection_accepts_unannotated_task_without_training_claim(self):
        self.meta["task_name"] = "Diagnostic trial: task not annotated"
        (self.raw / "meta.json").write_text(json.dumps(self.meta))
        with self.assertRaisesRegex(ValueError, "diagnostic placeholder"):
            export([self.raw], self.root / "training")
        output = self.root / "inspection"
        result = export([self.raw], output, inspection_only=True)
        self.assertEqual(result["status"], "INSPECTION_EXPORT_ONLY")
        self.assertFalse(result["training_ready"])
        self.assertEqual(result["frames"], 120)
        self.assertEqual(result["training_sequences"], 1)
        contract = json.loads((output / "meta/elf3_contract.json").read_text())
        self.assertFalse(contract["training_ready"])
        task = json.loads((output / "meta/tasks.jsonl").read_text())
        self.assertEqual(task["task"], self.meta["task_name"])
        source = json.loads((output / "meta/source_map.json").read_text())
        self.assertEqual(source["sequences"][0]["source_row_indices"], list(range(120)))

    def test_missing_head_is_accepted_body_is_required(self):
        for key in ("observation.joint_present", "action.joint_present"):
            rows = copy.deepcopy(self.rows)
            rows[1][key][29] = False
            write_rows(self.raw, rows)
            inspect_source(self.raw)
            rows[1][key][15] = False
            write_rows(self.raw, rows)
            with self.assertRaisesRegex(ValueError, "missing body"):
                inspect_source(self.raw)

    def test_discarded_and_wrong_order_refused(self):
        for change in ({"discarded": True},
                       {"joint_contract": {"canonical_names": RAW_JOINTS[:29] + ["left", "right"]}}):
            meta = {**self.meta, **change}
            (self.raw / "meta.json").write_text(json.dumps(meta))
            with self.assertRaises(ValueError):
                inspect_source(self.raw)

    def test_bad_token_and_reused_inference_refused(self):
        for key in ("token", "sequence"):
            rows = copy.deepcopy(self.rows)
            if key == "token":
                rows[1]["action.motion_token"][0] = .123
            else:
                rows[1]["sonic.inference"]["sequence"] = 0
            write_rows(self.raw, rows)
            with self.assertRaises(ValueError):
                inspect_source(self.raw)

    def test_missing_view_refused(self):
        (self.raw / self.rows[2]["observation.images"]["right_wrist_rgb"]).unlink()
        with self.assertRaisesRegex(ValueError, "JPEG"):
            inspect_source(self.raw)

    def test_gap_is_removed_and_real_rows_joined_without_filling(self):
        rows = [dict(t=100 + i / FPS + (1 if i >= 60 else 0), idx=i, segment=0)
                for i in range(120)]
        sequences, dropped = episode_sequences(rows)
        self.assertEqual([len(s) for s in sequences], [120])
        self.assertFalse(dropped)
        self.assertEqual(sequences[0][60]["idx"], 60)

    def test_short_sections_combine_into_one_training_window(self):
        rows = [dict(t=100 + i / FPS + (5 if i >= 20 else 0), idx=i, segment=int(i >= 20))
                for i in range(40)]
        sequences, dropped = episode_sequences(rows)
        self.assertEqual([len(x) for x in sequences], [40])
        self.assertFalse(dropped)
        self.assertEqual([x["idx"] for x in sequences[0]], list(range(40)))
        self.assertEqual(len(episode_sequences(rows[:39])[0][0]), 39)

    def test_distinct_recordings_remain_distinct_episodes(self):
        other = self.root / "another_episode"
        fixture(other)
        result = export([self.raw, other], self.root / "two_tasks")
        self.assertEqual(result["logical_recordings"], 2)
        self.assertEqual(result["training_sequences"], 2)
        self.assertEqual(result["frames"], 240)
        source = json.loads((self.root / "two_tasks/meta/source_map.json").read_text())
        self.assertEqual([s["source_id"] for s in source["sequences"]], [0, 1])
        self.assertEqual([len(s["source_row_indices"]) for s in source["sequences"]], [120, 120])

    def test_head_image_keeps_full_fov_without_stretch(self):
        image = np.full((1080, 1920, 3), 255, dtype=np.uint8)
        result = training_image(image)
        self.assertEqual(result.shape, (480, 640, 3))
        self.assertTrue((result[:60] == 0).all())
        self.assertTrue((result[60:420] == 255).all())
        self.assertTrue((result[420:] == 0).all())


    def test_wrong_units_or_motor_identity_refused(self):
        for field, value in (("position_unit", "deg"), ("bus", 6), ("motor_id", 2)):
            rows = copy.deepcopy(self.rows)
            rows[1]["gripper"]["native_raw_sample"]["samples"][0][field] = value
            write_rows(self.raw, rows)
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                inspect_source(self.raw)

    def test_30hz_metadata_and_relabelled_timestamps_refused(self):
        meta = {**self.meta, "fps": 30}
        (self.raw / "meta.json").write_text(json.dumps(meta))
        with self.assertRaisesRegex(ValueError, "50 Hz source"):
            inspect_source(self.raw)
        # Even changing only the label to 50 Hz must not produce training windows.
        rows = [dict(t=100 + i / 30, idx=i, segment=0) for i in range(120)]
        self.assertEqual(len(episode_sequences(rows)[0][0]), 120)  # Timing is reported, not used to delete rows.


if __name__ == "__main__":
    unittest.main()
