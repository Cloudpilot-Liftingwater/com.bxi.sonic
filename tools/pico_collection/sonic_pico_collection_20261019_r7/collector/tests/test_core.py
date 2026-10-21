from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import yaml

from scripts.state_machine_contract import ContractError, validate_sonic_controller

from attach_motion_tokens import validate_token_array
from check_config import validate_config
from export_lerobot_v21 import build_plan, export_dataset, frame_from_row
from real_robot_collection.contracts import (
    DIAGNOSTIC_ACTION_HISTORY,
    BODY_JOINT_NAMES,
    CANONICAL_JOINT_NAMES,
    PUBLISHED_FPS,
    SONIC_ACTION_SCALE,
    SONIC_DEFAULT_POSITION,
    canonicalize_named,
    effective_continuous_action_29,
    pack_action_31,
    pack_state_31,
    projected_gravity_wxyz,
    sonic_projected_gravity,
)
from real_robot_collection.pico import (
    HEADER_SIZE,
    RAW_REQUIRED_FIELDS,
    SMPL_REQUIRED_FIELDS,
    PicoSnapshot,
    decode_packed_message,
)
from real_robot_collection.recorder import RGB_CAMERA_NAMES, RealEpisodeRecorder
from real_robot_collection.state_machine import RecordingState
from real_robot_collection.synchronization import (
    CommandSample,
    SequenceResolver,
    SonicInferenceSample,
    matching_sonic_inference,
)
from validate_raw import read_rows, validate_episode


class ContractTests(unittest.TestCase):
    def test_sim_uses_an_isolated_ros_domain_by_default(self):
        common = (Path(__file__).resolve().parents[1] / "scripts" / "common.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn('BXI_SIM_ROS_DOMAIN_ID:-42', common)

    def test_controller_contract_rejects_legacy_short_state_graph(self):
        legacy = {
            "current": {"name": "normal"},
            "graph": {
                "states": [{"name": "normal"}, {"name": "sonic_teleop"}],
                "remote_events": {
                    "sonic_teleop_event": {"slot": "btn_10", "value": 7}
                },
            },
        }
        with self.assertRaisesRegex(ContractError, "legacy short-name"):
            validate_sonic_controller(legacy)

    def test_controller_contract_accepts_vla_sonic_graph(self):
        current = {
            "current": {"name": "com.bxi.basic_actions/normal"},
            "graph": {
                "states": [{"name": "com.bxi.sonic/sonic_teleop"}],
                "remote_events": {
                    "com.bxi.sonic/activate": {"slot": "btn_10", "value": 9}
                },
            },
        }
        summary = validate_sonic_controller(current)
        self.assertTrue(summary["compatible"])
        self.assertEqual(summary["activate"], "btn_10=9")

    def test_operator_shell_entrypoints_do_not_require_ripgrep(self):
        root = Path(__file__).resolve().parents[1]
        for name in ("start.sh", "sonic.sh", "control.sh", "deploy.sh"):
            script = (root / name).read_text(encoding="utf-8")
            self.assertNotIn(" rg ", script, name)
            self.assertNotIn("| rg", script, name)

    def test_sonic_button_command_always_generates_a_fresh_edge(self):
        script = (
            Path(__file__).resolve().parents[1] / "sonic.sh"
        ).read_text(encoding="utf-8")
        function = script.split("publish_button() {", 1)[1].split("\n}", 1)[0]
        neutral = 'communication/msg/MotionCommands "{}"'
        target = 'communication/msg/MotionCommands "{${button}}"'
        self.assertIn(neutral, function)
        self.assertIn(target, function)
        self.assertLess(function.index(neutral), function.index(target))

    def test_pico_mock_is_restricted_to_sim(self):
        script = (Path(__file__).resolve().parents[1] / "start.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn('SONIC_PICO_MOCK_VALUE="${SONIC_PICO_MOCK:-0}"', script)
        self.assertIn('if [[ "${PROFILE}" != "sim" ]]', script)
        self.assertIn("SYNTHETIC SIM-ONLY DIAGNOSTIC", script)

    def test_control_status_never_truncates_json_payload(self):
        script = (
            Path(__file__).resolve().parents[1] / "control.sh"
        ).read_text(encoding="utf-8")
        self.assertEqual(
            script.count("--full-length /elf3/recording_status"),
            2,
            "status and watch must both disable ros2 topic echo string truncation",
        )

    def test_control_finish_refuses_when_no_episode_is_recording(self):
        script = (
            Path(__file__).resolve().parents[1] / "control.sh"
        ).read_text(encoding="utf-8")
        self.assertIn('if [[ "${COMMAND}" == "finish" ]]', script)
        self.assertIn("no active recording", script)
        self.assertIn("Rows are created only after missing is empty", script)
        self.assertIn("remaining_rows_to_finish reaches 0", script)

    def test_interrupted_episode_is_preserved_outside_export_set(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = root / "config.yaml"
            config_path.write_text("mode: sim\n", encoding="utf-8")
            recorder = RealEpisodeRecorder(
                root / "source",
                mode="sim",
                fps=PUBLISHED_FPS,
                jpeg_quality=80,
                queue_capacity=8,
                rgb_camera_names=("head_rgb",),
                depth_enabled=False,
                camera_metadata={"head_rgb": {"serial": "sim"}},
                common_metadata={},
                config_path=config_path,
            )
            recorder.start_episode(
                {"task_id": "test", "instruction_text": "Test."},
                runtime_metadata={},
            )
            recorder.rows = [{"idx": 0, "timestamp": 0.0}]
            recorder.frame_count = 1
            with self.assertRaisesRegex(RuntimeError, "minimum is 100"):
                recorder.end_episode({"success": True}, 100)
            self.assertTrue(recorder.recording)
            result = recorder.preserve_interrupted_episode(
                "test_shutdown", min_frames=100
            )
            path = Path(result["path"])
            self.assertEqual(path.parent.name, "_interrupted")
            self.assertTrue((path / "frames.jsonl").is_file())
            metadata = json.loads((path / "meta.json").read_text(encoding="utf-8"))
            self.assertTrue(metadata["incomplete"])
            self.assertEqual(metadata["num_frames"], 1)
            self.assertEqual(metadata["termination"]["minimum_frames"], 100)
            self.assertFalse(any((root / "source").glob("episode_*")))

    def test_formal_config_requires_published_contract(self):
        path = Path(__file__).resolve().parents[1] / "config" / "real_collection.example.yaml"
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        self.assertEqual(config["dataset"]["fps"], PUBLISHED_FPS)
        self.assertFalse(config["requirements"]["require_camera_info"])
        self.assertTrue(config["gripper"]["enabled"])
        self.assertEqual(
            config["gripper"]["command_topic"],
            "/sonic/gripper_joint_commands",
        )
        self.assertEqual(config["cameras"]["head_depth"]["aligned_to"], "head_rgb")
        self.assertIn(
            "/aligned_depth_to_color/",
            config["cameras"]["head_depth"]["topic"],
        )
        self.assertEqual(
            config["metadata"]["camera_calibration_status"], "not_required"
        )
        for camera_name in (*RGB_CAMERA_NAMES, "head_depth"):
            self.assertEqual(
                config["cameras"][camera_name]["calibration_file"],
                "not_required",
            )
        self.assertEqual(config["motion_token"]["dimension"], 64)
        self.assertEqual(
            config["motion_token"]["sha256"],
            "50303783f72ca61674b54987cab20abd11d0ced25c7042062fb03ef6355cdfdf",
        )
        validate_config(config, formal=True, allow_placeholders=True)

    def test_named_canonicalization_supports_29_and_31(self):
        values = np.arange(29, dtype=np.float32)[::-1]
        output, mask = canonicalize_named(values, BODY_JOINT_NAMES[::-1], label="test")
        np.testing.assert_array_equal(output[:29], np.arange(29, dtype=np.float32))
        self.assertTrue(mask[:29].all())
        self.assertFalse(mask[29:].any())

    def test_33d_pack_contract(self):
        body = np.arange(29, dtype=np.float64)
        state = pack_state_31(body, [1.0, 2.0, 3.0, 4.0])
        action = pack_action_31(body, [0.01, 0.02])
        self.assertEqual(state.shape, (31,))
        np.testing.assert_array_equal(state[29:], [1.0, 2.0, 3.0, 4.0])
        np.testing.assert_array_equal(action[29:], [0.01, 0.01, 0.02, 0.02])

    def test_effective_action_is_diagnostic_29d(self):
        expected = np.linspace(-1.0, 1.0, 29, dtype=np.float32)
        target = SONIC_DEFAULT_POSITION + expected * SONIC_ACTION_SCALE
        actual = effective_continuous_action_29(target[::-1], BODY_JOINT_NAMES[::-1])
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=2.0e-6)
        self.assertEqual(actual.shape, (29,))

    def test_motion_tokens_require_64d_fsq_output(self):
        valid = np.zeros((3, 64), dtype=np.float64)
        self.assertEqual(validate_token_array(valid, 3).dtype, np.float64)
        with self.assertRaises(ValueError):
            validate_token_array(np.zeros((3, 29)), 3)
        valid[0, 0] = 0.01
        with self.assertRaises(ValueError):
            validate_token_array(valid, 3)

    def test_gravity_contract(self):
        np.testing.assert_allclose(projected_gravity_wxyz([1, 0, 0, 0]), [0, 0, -1])
        np.testing.assert_allclose(
            sonic_projected_gravity([1, 0, 0, 0], np.zeros(29)), [0, 0, -1]
        )

    def test_sequence_modes(self):
        resolver = SequenceResolver("frame_id_or_counter", "elf3/state/", -1)
        self.assertEqual(resolver.sequence("robot"), 0)
        self.assertEqual(resolver.sequence("elf3/state/20"), 20)
        self.assertEqual(resolver.sequence("robot"), 21)

    def test_sonic_bundle_matches_the_same_motor_target(self):
        position = np.arange(len(CANONICAL_JOINT_NAMES), dtype=np.float32)
        command = CommandSample(
            2.01, 10.0, 4, position, np.zeros(33), np.zeros(33),
            np.ones(33), np.ones(33), np.ones(33, dtype=bool),
            np.zeros(29), np.zeros((DIAGNOSTIC_ACTION_HISTORY, 29)),
        )
        bundle = SonicInferenceSample(
            2.0, 10.0, 7, "model", "a" * 64, position.copy(),
            np.zeros(64), "live_reference",
        )
        self.assertIs(
            matching_sonic_inference([bundle], command, 2.02, 0.05, 1.0e-5),
            bundle,
        )
        mismatched = SonicInferenceSample(
            2.0, 10.0, 8, "model", "a" * 64, position + 0.1,
            np.zeros(64), "live_reference",
        )
        self.assertIsNone(
            matching_sonic_inference([mismatched], command, 2.02, 0.05, 1.0e-5)
        )


class PicoAndStateTests(unittest.TestCase):
    def test_current_pico_boolean_protocol(self):
        header = {
            "endian": "le",
            "fields": [
                {"name": "left_trigger", "dtype": "f32", "shape": [1]},
                {"name": "right_trigger", "dtype": "f32", "shape": [1]},
                {"name": "toggle_data_collection", "dtype": "bool", "shape": [1]},
                {"name": "toggle_data_abort", "dtype": "bool", "shape": [1]},
                {"name": "start_data_collection", "dtype": "bool", "shape": [1]},
                {"name": "finish_data_collection", "dtype": "bool", "shape": [1]},
                {"name": "calibration_ready", "dtype": "bool", "shape": [1]},
                {"name": "stream_mode", "dtype": "i32", "shape": [1]},
            ],
        }
        encoded = json.dumps(header).encode()
        message = (
            b"pose" + encoded + bytes(HEADER_SIZE - len(encoded))
            + np.asarray([0.25], dtype="<f4").tobytes()
            + np.asarray([0.75], dtype="<f4").tobytes()
            + np.asarray([True], dtype=np.bool_).tobytes()
            + np.asarray([False], dtype=np.bool_).tobytes()
            + np.asarray([True], dtype=np.bool_).tobytes()
            + np.asarray([False], dtype=np.bool_).tobytes()
            + np.asarray([True], dtype=np.bool_).tobytes()
            + np.asarray([1], dtype="<i4").tobytes()
        )
        snapshot = PicoSnapshot(decode_packed_message(message, "pose"), 1.0)
        self.assertTrue(snapshot.toggle_collection)
        self.assertFalse(snapshot.abort)
        # Historical fields stay decodable but have no command properties.
        self.assertTrue(snapshot.fields["start_data_collection"].item())
        self.assertFalse(snapshot.fields["finish_data_collection"].item())
        self.assertEqual(snapshot.triggers, (0.25, 0.75))

    def test_toggle_state_machine(self):
        state = RecordingState()
        self.assertEqual(state.toggle(), "start")
        self.assertTrue(state.start())
        self.assertEqual(state.toggle(), "finish")
        self.assertTrue(state.finish())


def _pico_packet() -> dict:
    packet = {name: [0.0] for name in RAW_REQUIRED_FIELDS}
    packet.update(
        smpl_pose=np.zeros((10, 21, 3)).tolist(),
        smpl_joints=np.zeros((10, 24, 3)).tolist(),
        body_quat_w=np.zeros((10, 4)).tolist(),
        joint_pos=np.zeros((10, 29)).tolist(),
        joint_vel=np.zeros((10, 29)).tolist(),
        frame_index=list(range(10)), calibration_ready=[True], stream_mode=[1],
    )
    return packet


def _smpl_packet() -> dict:
    packet = {name: [0.0] for name in SMPL_REQUIRED_FIELDS}
    packet.update(
        term1_local=np.zeros((10, 72)).tolist(), root_quat=np.zeros((10, 4)).tolist(),
        wrist=np.zeros((10, 6)).tolist(), frame_index=[10], source_ready=[True],
        source_stream_mode=[1], source_calibration_ready=[True],
    )
    return packet


class RecorderAndExporterTests(unittest.TestCase):
    def test_formal_raw_to_strict_export_plan(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = root / "config.yaml"
            config_path.write_text("mode: real\n", encoding="utf-8")
            source = root / "lerobot_source"
            recorder = RealEpisodeRecorder(
                source, mode="real", fps=PUBLISHED_FPS, jpeg_quality=80, queue_capacity=128,
                rgb_camera_names=RGB_CAMERA_NAMES, depth_enabled=True,
                camera_metadata={name: {"serial": name} for name in RGB_CAMERA_NAMES},
                common_metadata={"camera_calibration_status": "not_required"},
                config_path=config_path,
            )
            recorder.start_episode(
                {"task_id": "test", "instruction_text": "Test the recorder."},
                runtime_metadata={"camera_info": {}},
            )
            joint_count = len(CANONICAL_JOINT_NAMES)
            state = {
                "position": np.arange(joint_count), "velocity": np.zeros(joint_count),
                "effort": np.zeros(joint_count), "driver_temperature": np.zeros(joint_count),
                "motor_temperature": np.zeros(joint_count), "joint_present": np.ones(joint_count, bool),
                "driver_temperature_present": np.ones(joint_count, bool),
                "motor_temperature_present": np.ones(joint_count, bool),
            }
            action = {
                "position": np.arange(joint_count), "velocity": np.zeros(joint_count),
                "torque": np.zeros(joint_count), "kp": np.ones(joint_count),
                "kd": np.ones(joint_count), "joint_present": np.ones(joint_count, bool),
            }
            images = {name: np.full((480, 640, 3), 40 + i, np.uint8) for i, name in enumerate(RGB_CAMERA_NAMES)}
            for index in range(100):
                recorder.capture(
                    anchor_sequence=index, observation=state, action=action,
                    effective_continuous_action_29=np.zeros(29),
                    diagnostic_action_history=np.zeros((DIAGNOSTIC_ACTION_HISTORY, 29)),
                    motion_token=np.zeros(64),
                    sonic_inference={
                        "sequence": index,
                        "model_id": "elf3_sonic_step28800_action_post_fsq",
                        "model_sha256": "a" * 64,
                        "policy_status": "live_reference",
                        "target_position": np.arange(joint_count).tolist(),
                    },
                    projected_gravity=[0, 0, -1],
                    imu={"orientation_wxyz": [1, 0, 0, 0], "angular_velocity": [0, 0, 0], "linear_acceleration": [0, 0, -9.81], "frame_id": "imu"},
                    rgb_images=images, depth_image=np.full((480, 640), 1000, np.uint16),
                    depth_encoding="16UC1", pico=_pico_packet(), smpl_ref=_smpl_packet(),
                    state_machine={"current": {"name": "com.bxi.sonic/sonic_teleop"}},
                    motion_command=None,
                    auxiliary={"selected_camera_sequences": {name: index for name in RGB_CAMERA_NAMES}},
                    gripper=_native_gripper_fixture(),
                    source_timestamps={"head_rgb": 1.0 + index/PUBLISHED_FPS},
                    source_receive_offsets={"state_minus_tick": -0.001},
                )
            episode = Path(recorder.end_episode({"success": True}, 100)["path"])
            self.assertEqual(validate_episode(episode, False, formal=True)["frames"], 100)
            plan = build_plan(source, formal=True)
            self.assertEqual(plan.total_frames, 100)
            self.assertEqual(plan.sonic_model_sha256, "a" * 64)
            self.assertEqual(set(plan.lerobot_features()), {
                "observation.state", "observation.projected_gravity", "action.wbc",
                "action.motion_token", "timestamp", "frame_index", "episode_index",
                "index", "task_index", "observation.images.ego_view",
                "observation.images.left_wrist", "observation.images.right_wrist",
            })
            frame = frame_from_row(
                episode, read_rows(episode / "frames.jsonl")[0], plan,
                motion_token=plan.episodes[0].motion_tokens[0],
            )
            self.assertEqual(frame["observation.state"].shape, (31,))
            self.assertEqual(frame["action.motion_token"].shape, (64,))

            count = 100
            arrays = {
                "observation.state": np.zeros((count, 33)),
                "observation.projected_gravity": np.tile([0, 0, -1], (count, 1)),
                "action.wbc": np.zeros((count, 33)),
                "action.motion_token": np.zeros((count, 64)),
                "timestamp": np.arange(count, dtype=np.float32) / PUBLISHED_FPS,
                "frame_index": np.arange(count, dtype=np.int64),
                "episode_index": np.zeros(count, dtype=np.int64),
                "index": np.arange(count, dtype=np.int64),
                "task_index": np.zeros(count, dtype=np.int64),
            }
            output = root / "lerobot_v21"
            export_dataset(plan, output=output)
            info = json.loads((output / "meta" / "info.json").read_text())
            self.assertEqual(info["robot_type"], "elf3_sonic_full_body")
            self.assertEqual(info["fps"], PUBLISHED_FPS)

            import pyarrow.parquet as pq
            table = pq.read_table(output/'data/chunk-000/episode_000000.parquet')
            self.assertEqual(table.schema.field('observation.state').type.list_size, 31)
            self.assertEqual(table.schema.field('action.wbc').type.list_size, 31)
            contract=json.loads((output/'meta/native66_contract.json').read_text())
            self.assertEqual(contract['model_action'],66)


def _native_gripper_fixture():
    from test_native66 import fixture
    from real_robot_collection.native66 import convert_sample
    mapping,raw=fixture(); state,command=convert_sample(raw,mapping,10.)
    return dict(trigger_intent=[0,0],state=state.tolist(),command=command.tolist(),
                state_is_measured=True,command_is_final=True,mapping=mapping,native_raw_sample=raw)


if __name__ == "__main__":
    unittest.main()
