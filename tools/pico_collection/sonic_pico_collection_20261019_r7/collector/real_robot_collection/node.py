#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import deque
import json
import os
from pathlib import Path
import sys
import threading
import time
from typing import Any

import numpy as np
import yaml

import rclpy
from nav_msgs.msg import Odometry
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, CompressedImage, Image, Imu, JointState
from std_msgs.msg import String

from communication.msg import (
    ActuatorCmds,
    ActuatorStates,
    MotionCommands,
    TouchSensor,
)

from .contracts import (
    DIAGNOSTIC_ACTION_HISTORY,
    BODY_JOINT_NAMES,
    CANONICAL_JOINT_NAMES,
    GRIPPER_COMMAND_NAMES,
    GRIPPER_STATE_JOINT_NAMES,
    PUBLISHED_FPS,
    TaskSpec,
    canonicalize_named,
    canonicalize_optional_named,
    effective_continuous_action_29,
    finite_vector,
    reorder_named,
    sonic_projected_gravity,
    tasks_from_config,
    validate_motion_token,
)
from .pico import PicoSnapshot, PicoSubscriber, PackedSubscriber, smpl_subscriber, smpl_reference_ready
from .recorder import DEPTH_CAMERA_NAME, RGB_CAMERA_NAMES, RealEpisodeRecorder
from .state_machine import RecordingState
from .native66 import validate_mapping, convert_sample
from .wrist_diagnostic import enabled as wrist_diagnostic_enabled, convert_raw
from .synchronization import (
    CommandSample,
    GripperSample,
    ImageSample,
    ImuSample,
    JsonSample,
    SequenceResolver,
    SonicInferenceSample,
    StateSample,
    decode_compressed_image,
    decode_depth_image,
    decode_raw_image,
    latest_at_or_before,
    matching_sonic_inference,
    quaternion_multiply_wxyz,
)


def _stamp_to_float(stamp) -> float:
    value = float(stamp.sec) + float(stamp.nanosec) * 1.0e-9
    if not np.isfinite(value) or value <= 0 or not 0 <= stamp.nanosec < 1000000000:
        raise ValueError('missing/invalid source timestamp; receipt time is not a substitute')
    return value


def _json_stamp_to_float(value: object) -> float:
    if not isinstance(value, dict):
        raise ValueError("stamp must be an object")
    stamp = float(value.get("sec", 0)) + float(value.get("nanosec", 0)) * 1.0e-9
    if not np.isfinite(stamp) or stamp <= 0.0:
        raise ValueError("stamp must be a positive finite ROS time")
    return stamp


def _require_mapping(config: dict, key: str) -> dict:
    value = config.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"configuration section {key!r} must be a mapping")
    return value


def _positive_seconds(mapping: dict, key: str, default_ms: float) -> float:
    value = float(mapping.get(key, default_ms)) / 1000.0
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{key} must be positive and finite")
    return value


def _vector3(value) -> list[float]:
    return [float(value.x), float(value.y), float(value.z)]


def _camera_callback(method, name):
    def callback(msg):
        return method(name, msg)
    return callback


def _camera_info_dict(msg: CameraInfo) -> dict[str, Any]:
    return {
        "stamp_unix": _stamp_to_float(msg.header.stamp),
        "frame_id": str(msg.header.frame_id),
        "height": int(msg.height),
        "width": int(msg.width),
        "distortion_model": str(msg.distortion_model),
        "d": [float(x) for x in msg.d],
        "k": [float(x) for x in msg.k],
        "r": [float(x) for x in msg.r],
        "p": [float(x) for x in msg.p],
        "binning_x": int(msg.binning_x),
        "binning_y": int(msg.binning_y),
    }


class RealCollectionNode(Node):
    def __init__(self, config_path: Path, data_id: str | None) -> None:
        super().__init__("elf3_real_collection")
        self.config_path = Path(config_path).resolve()
        with self.config_path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        if not isinstance(config, dict):
            raise ValueError("configuration root must be a mapping")
        self.config = config
        self.wrist_diagnostic = wrist_diagnostic_enabled(config)
        self.mode = str(config.get("mode", "")).strip()
        if self.mode not in ("sim", "hybrid_debug", "real"):
            raise ValueError("mode must be sim, hybrid_debug, or real")
        self.dataset_cfg = _require_mapping(config, "dataset")
        self.robot_cfg = _require_mapping(config, "robot")
        self.sequence_cfg = _require_mapping(config, "sequence")
        self.gripper_cfg = _require_mapping(config, "gripper")
        if not self.wrist_diagnostic:
            validate_mapping(self.gripper_cfg.get("mapping"))
        self.motion_token_cfg = _require_mapping(config, "motion_token")
        self.pico_cfg = _require_mapping(config, "pico")
        self.smpl_cfg = _require_mapping(config, "smpl_ref")
        self.camera_cfg = _require_mapping(config, "cameras")
        self.requirements_cfg = _require_mapping(config, "requirements")
        self.metadata_cfg = _require_mapping(config, "metadata")
        self.tasks = tasks_from_config(self.dataset_cfg)
        self.default_task_id = str(
            self.dataset_cfg.get("default_task_id", self.tasks[0].task_id)
        )
        if self._task(self.default_task_id) is None:
            raise ValueError(f"unknown dataset.default_task_id={self.default_task_id!r}")
        self.fps = int(self.dataset_cfg.get("fps", PUBLISHED_FPS))
        self.min_frames = int(self.dataset_cfg.get("min_episode_frames", 30))
        if self.fps <= 0 or self.min_frames <= 0:
            raise ValueError("dataset fps and min_episode_frames must be positive")
        expected_fps = (50 if self.wrist_diagnostic and
                        self.metadata_cfg.get('capture_profile') == 'sonic50_latest_rgb' else PUBLISHED_FPS)
        if self.fps != expected_fps:
            raise ValueError(
                f"dataset.fps must be {expected_fps} for the published data contract"
            )

        configured_names = tuple(str(x) for x in self.robot_cfg["body_joint_names"])
        if configured_names != BODY_JOINT_NAMES:
            raise ValueError(
                "robot.body_joint_names must exactly match the canonical 29D SONIC order"
            )
        self.sequence = SequenceResolver(
            str(self.sequence_cfg.get("mode", "counter")),
            str(self.sequence_cfg.get("frame_prefix", "elf3/state/")),
            int(self.sequence_cfg.get("counter_start", -1)),
        )
        self.history_size = int(self.sequence_cfg.get("history_size", 512))
        if self.history_size < DIAGNOSTIC_ACTION_HISTORY:
            raise ValueError("sequence.history_size must be at least 10")
        self.pairing_timeout_s = _positive_seconds(
            self.sequence_cfg, "pairing_timeout_ms", 250.0
        )
        self.max_robot_delta_s = _positive_seconds(
            self.robot_cfg, "max_robot_delta_ms", 50.0
        )
        self.max_imu_delta_s = _positive_seconds(
            self.robot_cfg, "max_imu_delta_ms", 50.0
        )
        self.max_state_machine_age_s = _positive_seconds(
            self.robot_cfg, "max_state_machine_age_ms", 300.0
        )
        self.max_motion_token_delta_s = _positive_seconds(
            self.motion_token_cfg, "max_delta_ms", 50.0
        )
        self.motion_token_action_atol = float(
            self.motion_token_cfg.get("action_match_atol", 1.0e-5)
        )
        if (
            not np.isfinite(self.motion_token_action_atol)
            or self.motion_token_action_atol <= 0.0
        ):
            raise ValueError("motion_token.action_match_atol must be positive")
        self.motion_token_topic = str(self.motion_token_cfg.get("topic", "")).strip()
        if not self.motion_token_topic.startswith("/"):
            raise ValueError("motion_token.topic must be an absolute ROS topic")
        self.motion_token_model_id = str(
            self.motion_token_cfg.get(
                "model_id", "elf3_sonic_step28800_action_post_fsq"
            )
        )
        self.motion_token_model_sha256 = str(
            self.motion_token_cfg.get("sha256", "")
        ).lower()
        if len(self.motion_token_model_sha256) != 64 or any(
            character not in "0123456789abcdef"
            for character in self.motion_token_model_sha256
        ):
            raise ValueError("motion_token.sha256 must contain 64 lowercase hex digits")
        self.required_policy_status = str(
            self.motion_token_cfg.get("required_policy_status", "live_reference")
        )
        self.max_camera_delta_s = _positive_seconds(
            self.camera_cfg, "max_delta_ms", 50.0
        )
        self.max_pico_delta_s = _positive_seconds(
            self.pico_cfg, "max_delta_ms", 200.0
        )
        self.max_smpl_delta_s = _positive_seconds(
            self.smpl_cfg, "max_delta_ms", 200.0
        )
        self.imu_correction = finite_vector(
            self.robot_cfg.get(
                "imu_orientation_correction_wxyz", [1.0, 0.0, 0.0, 0.0]
            ),
            4,
            "robot.imu_orientation_correction_wxyz",
            dtype=np.float64,
        )

        self.camera_definitions: dict[str, dict] = {}
        for name in (*RGB_CAMERA_NAMES, DEPTH_CAMERA_NAME):
            definition = self.camera_cfg.get(name)
            if not isinstance(definition, dict):
                raise ValueError(f"cameras.{name} must be a mapping")
            self.camera_definitions[name] = definition
        self.active_rgb = tuple(
            name
            for name in RGB_CAMERA_NAMES
            if bool(self.camera_definitions[name].get("enabled", True))
        )
        if "head_rgb" not in self.active_rgb and not self.wrist_diagnostic:
            raise ValueError("head_rgb must be enabled because it is the sample anchor")
        self.depth_enabled = bool(
            self.camera_definitions[DEPTH_CAMERA_NAME].get("enabled", True)
        )

        output_root = Path(str(self.dataset_cfg.get("output_root", "data"))).expanduser()
        if not output_root.is_absolute():
            output_root = self.config_path.parent.parent / output_root
        if not data_id:
            data_id = f"elf3_{self.mode}_{time.strftime('%Y%m%d_%H%M%S')}"
        self.data_id = str(data_id)
        if Path(self.data_id).name != self.data_id or self.data_id in ("", ".", ".."):
            raise ValueError("DATA_ID must be one safe directory name")
        self.source_root = output_root / self.data_id / (
            "three_rgb_gripper_source" if self.wrist_diagnostic else "lerobot_source")
        camera_meta = {
            name: {
                key: value
                for key, value in definition.items()
                if key not in ("topic", "camera_info_topic", "message_type")
            }
            for name, definition in self.camera_definitions.items()
            if bool(definition.get("enabled", True))
        }
        common_meta = dict(self.metadata_cfg)
        common_meta.update(
            {
                "source": "elf3_real_robot_collection",
                "mode": self.mode,
                "topics": {
                    key: value
                    for key, value in self.robot_cfg.items()
                    if key.endswith("_topic")
                },
                "pico_endpoint": (
                    f"tcp://{self.pico_cfg['host']}:{self.pico_cfg['port']}"
                ),
                "smpl_ref_endpoint": (
                    f"tcp://{self.smpl_cfg['host']}:{self.smpl_cfg['port']}"
                ),
                "motion_token": dict(self.motion_token_cfg),
            }
        )
        recorder_type = RealEpisodeRecorder
        if self.metadata_cfg.get('storage_backend') == 'nvidia_sonic_lerobot_v1':
            from .official_recorder import OfficialEpisodeRecorder
            recorder_type = OfficialEpisodeRecorder
        self.recorder = recorder_type(
            self.source_root,
            mode=self.mode,
            fps=self.fps,
            jpeg_quality=int(self.dataset_cfg.get("jpeg_quality", 92)),
            queue_capacity=int(self.dataset_cfg.get("queue_capacity", 128)),
            rgb_camera_names=self.active_rgb,
            depth_enabled=self.depth_enabled,
            camera_metadata=camera_meta,
            common_metadata=common_meta,
            config_path=self.config_path,
            wrist_diagnostic=self.wrist_diagnostic,
        )

        self.lock = threading.RLock()
        self.states: deque[StateSample] = deque(maxlen=self.history_size)
        self.commands: deque[CommandSample] = deque(maxlen=self.history_size)
        self.sonic_inferences: deque[SonicInferenceSample] = deque(
            maxlen=self.history_size
        )
        self.imus: deque[ImuSample] = deque(maxlen=self.history_size)
        self.gripper_states: deque[GripperSample] = deque(maxlen=self.history_size)
        self.gripper_commands: deque[GripperSample] = deque(maxlen=self.history_size)
        self.images: dict[str, deque[ImageSample]] = {
            name: deque(maxlen=int(self.camera_cfg.get("history_size", 120)))
            for name in (*RGB_CAMERA_NAMES, DEPTH_CAMERA_NAME)
        }
        self.camera_counts = {name: -1 for name in self.images}
        self.camera_info: dict[str, dict[str, Any]] = {}
        self.diagnostic_actions: deque[np.ndarray] = deque(maxlen=DIAGNOSTIC_ACTION_HISTORY)
        self.state_count = -1
        self.command_count = -1
        self.last_received_inference_sequence = -1
        self.last_recorded_inference_sequence = -1
        self.latest_state_machine: JsonSample | None = None
        self.latest_motion_command: JsonSample | None = None
        self.latest_odom: JsonSample | None = None
        self.latest_touch: JsonSample | None = None
        self.last_anchor_sequence = -1
        self.episode_anchor_floor = -1
        self.episode_tick_count = 0
        self.last_selected_camera_sequences = {name: -1 for name in self.images}
        self.last_error = ""
        self.last_missing: list[str] = []
        self.dropped_control_ticks = 0
        self.first_row_logged = False
        self.closed = False
        self.manual_events: deque[dict[str, Any]] = deque()
        self.state_machine = RecordingState()
        self.last_pico_control_packet_time = 0.0

        self.input_callback_groups = {name: MutuallyExclusiveCallbackGroup()
                                      for name in ('state', 'command', 'imu', 'inference', 'status')}
        latest_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT,
                               durability=DurabilityPolicy.VOLATILE)
        sensor_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=4,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.create_subscription(
            ActuatorStates,
            str(self.robot_cfg["actuator_state_topic"]),
            self._on_state,
            latest_qos,
            callback_group=self.input_callback_groups['state'],
        )
        self.create_subscription(
            ActuatorCmds,
            str(self.robot_cfg["actuator_command_topic"]),
            self._on_command,
            latest_qos,
            callback_group=self.input_callback_groups['command'],
        )
        self.create_subscription(
            String,
            self.motion_token_topic,
            self._on_sonic_inference,
            QoSProfile(depth=256),
            callback_group=self.input_callback_groups['inference'],
        )
        self.create_subscription(
            Imu,
            str(self.robot_cfg["imu_topic"]),
            self._on_imu,
            latest_qos,
            callback_group=self.input_callback_groups['imu'],
        )
        self.create_subscription(
            String,
            str(self.robot_cfg["state_machine_topic"]),
            self._on_state_machine,
            10,
            callback_group=self.input_callback_groups['status'],
        )
        self.create_subscription(
            MotionCommands,
            str(self.robot_cfg.get("motion_commands_topic", "/motion_commands")),
            self._on_motion_command,
            sensor_qos,
        )
        if str(self.robot_cfg.get("odom_topic", "")).strip():
            self.create_subscription(
                Odometry,
                str(self.robot_cfg["odom_topic"]),
                self._on_odom,
                sensor_qos,
            )
        if str(self.robot_cfg.get("touch_topic", "")).strip():
            self.create_subscription(
                TouchSensor,
                str(self.robot_cfg["touch_topic"]),
                self._on_touch,
                sensor_qos,
            )
        self.gripper_enabled = bool(self.gripper_cfg.get("enabled", False))
        if (
            self.mode == "real"
            and bool(self.requirements_cfg.get("formal_collection", False))
            and not self.gripper_enabled
        ):
            raise ValueError(
                "formal real collection requires measured gripper state and final command"
            )
        if self.gripper_enabled:
            # In this buffered diagnostic, do not serialize gripper reception
            # behind image-row copying. Still process gripper callbacks in order.
            self.gripper_callback_group = MutuallyExclusiveCallbackGroup() if self.wrist_diagnostic else None
            gripper_qos = QoSProfile(depth=32, reliability=ReliabilityPolicy.BEST_EFFORT,
                durability=DurabilityPolicy.VOLATILE) if self.wrist_diagnostic else sensor_qos
            self.create_subscription(
                String,
                str(self.gripper_cfg["sample_topic"]),
                self._on_native_gripper,
                gripper_qos,
                callback_group=self.gripper_callback_group,
            )

        camera_callbacks = ReentrantCallbackGroup()
        for name in self.active_rgb:
            camera_callbacks = MutuallyExclusiveCallbackGroup()
            definition = self.camera_definitions[name]
            callback = _camera_callback(self._on_rgb, name)
            message_type = str(definition.get("message_type", "raw"))
            ros_type = CompressedImage if message_type == "compressed" else Image
            if message_type not in ("raw", "compressed"):
                raise ValueError(f"cameras.{name}.message_type must be raw or compressed")
            self.create_subscription(
                ros_type,
                str(definition["topic"]),
                callback,
                sensor_qos,
                callback_group=camera_callbacks,
            )
            self.create_subscription(
                CameraInfo,
                str(definition["camera_info_topic"]),
                _camera_callback(self._on_camera_info, name),
                sensor_qos,
                callback_group=camera_callbacks,
            )
        if self.depth_enabled:
            definition = self.camera_definitions[DEPTH_CAMERA_NAME]
            self.create_subscription(
                Image,
                str(definition["topic"]),
                self._on_depth,
                sensor_qos,
                callback_group=camera_callbacks,
            )
            self.create_subscription(
                CameraInfo,
                str(definition["camera_info_topic"]),
                lambda msg: self._on_camera_info(DEPTH_CAMERA_NAME, msg),
                sensor_qos,
                callback_group=camera_callbacks,
            )

        self.status_publisher = self.create_publisher(
            String, "/elf3/recording_status", 10
        )
        self.create_subscription(
            String, "/elf3/recording_control", self._on_control, 10
        )
        self.create_timer(1.0 / float(self.fps), self._capture_tick)
        self.create_timer(0.005, self._control_tick)
        self.create_timer(1.0, self._publish_status)

        self.pico = PicoSubscriber(
            str(self.pico_cfg.get("host", "127.0.0.1")),
            int(self.pico_cfg.get("port", 5556)),
            str(self.pico_cfg.get("topic", "pose")),
            history_size=min(self.history_size, 256),
        )
        self.smpl: PackedSubscriber = smpl_subscriber(
            str(self.smpl_cfg.get("host", "127.0.0.1")),
            int(self.smpl_cfg.get("port", 5557)),
            str(self.smpl_cfg.get("topic", "smpl_ref")),
            history_size=min(self.history_size, 256),
        )
        self.pico.start()
        self.smpl.start()
        self.get_logger().info(
            f"ready mode={self.mode} data_id={self.data_id} source={self.source_root}"
        )

    def _task(self, task_id: str) -> TaskSpec | None:
        return next((task for task in self.tasks if task.task_id == task_id), None)

    def _on_state(self, msg: ActuatorStates) -> None:
        now = time.monotonic()
        try:
            position, present = canonicalize_named(
                msg.position, msg.name, label="actuator state position"
            )
            velocity, _ = canonicalize_named(
                msg.velocity, msg.name, label="actuator state velocity"
            )
            effort, _ = canonicalize_named(
                msg.effort, msg.name, label="actuator state effort"
            )
            driver_temp, driver_present = canonicalize_optional_named(
                msg.driver_temperature,
                msg.name,
                present_names=(),
                label="driver temperature",
            )
            motor_temp, motor_present = canonicalize_optional_named(
                msg.motor_temperature,
                msg.name,
                present_names=(),
                label="motor temperature",
            )
            self.state_count += 1
            sequence = self.sequence.sequence(msg.header.frame_id)
            sample = StateSample(
                now,
                _stamp_to_float(msg.header.stamp),
                sequence,
                position,
                velocity,
                effort,
                driver_temp,
                motor_temp,
                present,
                driver_present,
                motor_present,
            )
        except Exception as exc:
            self._set_error(f"invalid actuator state: {exc}")
            return
        with self.lock:
            self.states.append(sample)

    def _on_command(self, msg: ActuatorCmds) -> None:
        now = time.monotonic()
        try:
            position, present = canonicalize_named(
                msg.pos, msg.actuators_name, label="actuator command position"
            )
            velocity, _ = canonicalize_named(
                msg.vel, msg.actuators_name, label="actuator command velocity"
            )
            torque, _ = canonicalize_named(
                msg.torque, msg.actuators_name, label="actuator command torque"
            )
            kp, _ = canonicalize_named(
                msg.kp, msg.actuators_name, label="actuator command kp"
            )
            kd, _ = canonicalize_named(
                msg.kd, msg.actuators_name, label="actuator command kd"
            )
            token = effective_continuous_action_29(msg.pos, msg.actuators_name)
            with self.lock:
                self.diagnostic_actions.append(token.copy())
                history = (
                    np.stack(tuple(self.diagnostic_actions), axis=0).astype(np.float32)
                    if len(self.diagnostic_actions) == DIAGNOSTIC_ACTION_HISTORY
                    else None
                )
                self.command_count += 1
                sequence = self.command_count
                self.commands.append(
                    CommandSample(
                        now,
                        _stamp_to_float(msg.header.stamp),
                        sequence,
                        position,
                        velocity,
                        torque,
                        kp,
                        kd,
                        present,
                        token,
                        history,
                    )
                )
        except Exception as exc:
            self._set_error(f"invalid actuator command: {exc}")

    def _on_sonic_inference(self, msg: String) -> None:
        now = time.monotonic()
        try:
            value = json.loads(msg.data)
            if not isinstance(value, dict) or int(value.get("schema", -1)) != 1:
                raise ValueError("unsupported inference result schema")
            if value.get("motion_token_valid") is not True:
                raise ValueError("motion_token_valid is not true")
            sequence = int(value.get("inference_sequence", -1))
            if sequence < 0:
                raise ValueError("inference_sequence must be non-negative")
            model_id = str(value.get("model_id", ""))
            if model_id != self.motion_token_model_id:
                raise ValueError(
                    f"model_id={model_id!r}, expected {self.motion_token_model_id!r}"
                )
            model_sha256 = str(value.get("model_sha256", "")).lower()
            if len(model_sha256) != 64 or any(
                character not in "0123456789abcdef" for character in model_sha256
            ):
                raise ValueError("model_sha256 must contain 64 lowercase hex digits")
            if model_sha256 != self.motion_token_model_sha256:
                raise ValueError(
                    f"model_sha256={model_sha256!r}, expected "
                    f"{self.motion_token_model_sha256!r}"
                )
            target_position, present = canonicalize_named(
                value.get("target_position"),
                value.get("joint_names", ()),
                label="SONIC inference target",
            )
            if not present[: len(BODY_JOINT_NAMES)].all():
                raise ValueError("SONIC inference target is missing body joints")
            token = validate_motion_token(value.get("motion_token"))
            policy_status = str(value.get("policy_status", ""))
            if policy_status != self.required_policy_status:
                raise ValueError(
                    f"policy_status={policy_status!r}, expected "
                    f"{self.required_policy_status!r}"
                )
            sample = SonicInferenceSample(
                now,
                _json_stamp_to_float(value.get("stamp")),
                sequence,
                model_id,
                model_sha256,
                target_position,
                token,
                policy_status,
            )
            with self.lock:
                if sequence <= self.last_received_inference_sequence:
                    return
                if self.sonic_inferences and (
                    self.sonic_inferences[-1].model_sha256 != model_sha256
                ):
                    raise ValueError("Sonic model checksum changed while collector is running")
                self.last_received_inference_sequence = sequence
                self.sonic_inferences.append(sample)
                if (
                    self.last_error is not None
                    and self.last_error.startswith("invalid Sonic inference result:")
                ):
                    self.last_error = None
        except Exception as exc:
            self._set_error(f"invalid Sonic inference result: {exc}")

    def _on_imu(self, msg: Imu) -> None:
        now = time.monotonic()
        try:
            q = quaternion_multiply_wxyz(
                [
                    msg.orientation.w,
                    msg.orientation.x,
                    msg.orientation.y,
                    msg.orientation.z,
                ],
                self.imu_correction,
            )
            angular = finite_vector(
                _vector3(msg.angular_velocity), 3, "IMU angular velocity"
            )
            linear = finite_vector(
                _vector3(msg.linear_acceleration), 3, "IMU linear acceleration"
            )
            sample = ImuSample(
                now,
                _stamp_to_float(msg.header.stamp),
                q,
                angular,
                linear,
                str(msg.header.frame_id),
            )
        except Exception as exc:
            self._set_error(f"invalid IMU: {exc}")
            return
        with self.lock:
            self.imus.append(sample)

    def _on_native_gripper(self, msg: String) -> None:
        try:
            received = time.monotonic()
            value = json.loads(msg.data)
            state, command = (convert_raw(value, received, self.gripper_cfg['expected_identities']) if self.wrist_diagnostic
                              else convert_sample(value, self.gripper_cfg["mapping"], time.monotonic()))
            stamp = _json_stamp_to_float(value["stamp"])
            source_time = float(value["monotonic"])
            with self.lock:
                self.gripper_states.append(GripperSample(received, stamp, state, value))
                self.gripper_commands.append(GripperSample(received, stamp, command, value))
        except Exception as exc:
            with self.lock:
                self.gripper_states.clear()
                self.gripper_commands.clear()
            self._set_error(f"invalid native66 gripper sample: {exc}")

    def _on_gripper_state(self, msg: JointState) -> None:
        try:
            names = tuple(str(x) for x in self.gripper_cfg["state_joint_names"])
            if len(names) != len(GRIPPER_STATE_JOINT_NAMES):
                raise ValueError("gripper.state_joint_names must contain two joints")
            values = reorder_named(msg.position, msg.name, names, "gripper state")
            sample = GripperSample(
                time.monotonic(), _stamp_to_float(msg.header.stamp), values
            )
        except Exception as exc:
            self._set_error(f"invalid gripper state: {exc}")
            return
        with self.lock:
            self.gripper_states.append(sample)

    def _on_gripper_command(self, msg: JointState) -> None:
        """Capture the final command submitted to the two independent grippers."""

        try:
            names = tuple(str(x) for x in self.gripper_cfg["command_joint_names"])
            if len(names) != len(GRIPPER_COMMAND_NAMES):
                raise ValueError("gripper.command_joint_names must contain left and right")
            values = reorder_named(msg.position, msg.name, names, "gripper command")
            sample = GripperSample(
                time.monotonic(), _stamp_to_float(msg.header.stamp), values
            )
        except Exception as exc:
            self._set_error(f"invalid gripper command: {exc}")
            return
        with self.lock:
            self.gripper_commands.append(sample)

    def _on_rgb(self, name: str, msg: Image | CompressedImage) -> None:
        now = time.monotonic()
        try:
            rgb = (
                decode_compressed_image(bytes(msg.data))
                if isinstance(msg, CompressedImage)
                else decode_raw_image(msg)
            )
            stamp = _stamp_to_float(msg.header.stamp)
            frame_id = str(msg.header.frame_id)
        except Exception as exc:
            self._set_error(f"invalid {name}: {exc}")
            return
        with self.lock:
            self.camera_counts[name] += 1
            self.images[name].append(
                ImageSample(
                    now,
                    stamp,
                    self.camera_counts[name],
                    rgb,
                    "rgb8",
                    frame_id,
                )
            )

    def _on_depth(self, msg: Image) -> None:
        now = time.monotonic()
        try:
            depth, encoding = decode_depth_image(msg)
        except Exception as exc:
            self._set_error(f"invalid head depth: {exc}")
            return
        with self.lock:
            self.camera_counts[DEPTH_CAMERA_NAME] += 1
            self.images[DEPTH_CAMERA_NAME].append(
                ImageSample(
                    now,
                    _stamp_to_float(msg.header.stamp),
                    self.camera_counts[DEPTH_CAMERA_NAME],
                    depth,
                    encoding,
                    str(msg.header.frame_id),
                )
            )

    def _on_camera_info(self, name: str, msg: CameraInfo) -> None:
        try:
            value = _camera_info_dict(msg)
        except Exception as exc:
            self._set_error(f"invalid {name} CameraInfo: {exc}")
            return
        with self.lock:
            self.camera_info[name] = value

    def _on_state_machine(self, msg: String) -> None:
        try:
            value = json.loads(msg.data)
            if not isinstance(value, dict):
                raise ValueError("state-machine status must be a JSON object")
            stamp = value.get("stamp", {})
            stamp_unix = float(stamp.get("sec", 0)) + float(
                stamp.get("nanosec", 0)
            ) * 1.0e-9
            sample = JsonSample(time.monotonic(), stamp_unix or time.time(), value)
        except Exception as exc:
            self._set_error(f"invalid state-machine status: {exc}")
            return
        with self.lock:
            previous_name = (
                str(
                    self.latest_state_machine.value.get("current", {}).get(
                        "name", ""
                    )
                )
                if self.latest_state_machine is not None
                else ""
            )
            current_name = str(value.get("current", {}).get("name", ""))
            if (
                current_name == self._required_state_name()
                and previous_name != current_name
            ):
                # The model's 10-step history is reset on SONIC entry. Do not
                # label preceding normal-state commands as SONIC action tokens.
                self.diagnostic_actions.clear()
                self.commands.clear()
            self.latest_state_machine = sample

    def _on_motion_command(self, msg: MotionCommands) -> None:
        value = {
            "stamp_unix": _stamp_to_float(msg.header.stamp),
            "vel_des": _vector3(msg.vel_des),
            "height_des": float(msg.height_des),
            "yawdot_des": float(msg.yawdot_des),
            "mode": int(msg.mode),
            "buttons": [int(getattr(msg, f"btn_{i}")) for i in range(1, 11)],
            "axes": [int(getattr(msg, f"axis_{i}")) for i in range(1, 11)],
        }
        with self.lock:
            self.latest_motion_command = JsonSample(
                time.monotonic(), value["stamp_unix"], value
            )

    def _on_odom(self, msg: Odometry) -> None:
        value = {
            "frame_id": str(msg.header.frame_id),
            "child_frame_id": str(msg.child_frame_id),
            "position": _vector3(msg.pose.pose.position),
            "orientation_wxyz": [
                float(msg.pose.pose.orientation.w),
                float(msg.pose.pose.orientation.x),
                float(msg.pose.pose.orientation.y),
                float(msg.pose.pose.orientation.z),
            ],
            "linear_velocity": _vector3(msg.twist.twist.linear),
            "angular_velocity": _vector3(msg.twist.twist.angular),
        }
        with self.lock:
            self.latest_odom = JsonSample(
                time.monotonic(), _stamp_to_float(msg.header.stamp), value
            )

    def _on_touch(self, msg: TouchSensor) -> None:
        value = {
            "frame_id": str(msg.header.frame_id),
            "value": [float(x) for x in msg.value],
        }
        with self.lock:
            self.latest_touch = JsonSample(
                time.monotonic(), _stamp_to_float(msg.header.stamp), value
            )

    def _on_control(self, msg: String) -> None:
        try:
            event = json.loads(msg.data)
            commands = {"start", "finish", "abort", "rearm"}
            if not isinstance(event, dict) or str(event.get("command", "")) not in commands:
                raise ValueError(f"command must be one of {sorted(commands)}")
            with self.lock:
                self.manual_events.append(event)
        except Exception as exc:
            self._set_error(f"invalid /elf3/recording_control JSON: {exc}")

    def _current_pico(self, now: float) -> PicoSnapshot | None:
        snapshot = self.pico.latest()
        if not isinstance(snapshot, PicoSnapshot):
            return None
        return (
            snapshot
            if now - snapshot.received_monotonic <= self.max_pico_delta_s
            else None
        )

    def _required_state_name(self) -> str:
        return str(
            self.requirements_cfg.get(
                "state_machine_state", "com.bxi.sonic/sonic_teleop"
            )
        )

    @staticmethod
    def _smpl_ready(snapshot) -> bool:
        return smpl_reference_ready(snapshot)

    def _ready(self, now: float) -> list[str]:
        missing: list[str] = []
        with self.lock:
            if not self.states:
                missing.append("actuator_state")
            if not self.commands:
                missing.append("actuator_command")
            elif self.commands[-1].diagnostic_action_history is None:
                missing.append("diagnostic_action_history_10")
            if not self.imus:
                missing.append("imu")
            if not self.sonic_inferences:
                missing.append("sonic_inference_result")
            elif (
                now - self.sonic_inferences[-1].received_monotonic
                > self.max_motion_token_delta_s
            ):
                missing.append("fresh_sonic_inference_result")
            for name in self.active_rgb:
                if not self.images[name]:
                    missing.append(name)
            if self.depth_enabled and not self.images[DEPTH_CAMERA_NAME]:
                missing.append(DEPTH_CAMERA_NAME)
            if bool(self.requirements_cfg.get("require_camera_info", False)):
                for name in (*self.active_rgb,):
                    info = self.camera_info.get(name)
                    if info is None:
                        missing.append(f"{name}_camera_info")
                    elif self.mode == "real" and not any(info.get("k", ())):
                        missing.append(f"{name}_calibration")
                if self.depth_enabled:
                    depth_info = self.camera_info.get(DEPTH_CAMERA_NAME)
                    if depth_info is None:
                        missing.append("head_depth_camera_info")
                    elif self.mode == "real" and not any(depth_info.get("k", ())):
                        missing.append("head_depth_calibration")
            status = self.latest_state_machine
        if status is None or now - status.received_monotonic > self.max_state_machine_age_s:
            missing.append("state_machine_status")
        else:
            current = status.value.get("current", {})
            if str(current.get("name", "")) != self._required_state_name():
                missing.append("sonic_state")
        pico = self._current_pico(now)
        if pico is None:
            missing.append("pico_raw")
        else:
            if bool(self.requirements_cfg.get("require_pico_calibrated", True)):
                if not pico.calibrated or pico.stream_mode != 1:
                    missing.append("pico_calibrated_pose_mode")
        smpl = self.smpl.latest()
        if smpl is None or now - smpl.received_monotonic > self.max_smpl_delta_s:
            missing.append("smpl_ref")
        elif not self._smpl_ready(smpl):
            missing.append("smpl_ref_ready")
        if self.gripper_enabled:
            with self.lock:
                if not self.gripper_states or now-self.gripper_states[-1].received_monotonic > self.max_robot_delta_s:
                    missing.append("gripper_state")
                if not self.gripper_commands or now-self.gripper_commands[-1].received_monotonic > self.max_robot_delta_s:
                    missing.append("gripper_command")
        return sorted(set(missing))

    def _runtime_metadata(self) -> dict[str, Any]:
        with self.lock:
            status = self.latest_state_machine.value if self.latest_state_machine else None
            camera_info = dict(self.camera_info)
        return {
            "camera_info": camera_info,
            "state_machine_at_start": status,
            "joint_names": list(CANONICAL_JOINT_NAMES),
            "gripper_feedback_enabled": self.gripper_enabled,
            "gripper_state_joint_names": list(
                self.gripper_cfg.get("state_joint_names", ())
            ),
            "gripper_command_joint_names": list(
                self.gripper_cfg.get("command_joint_names", ())
            ),
            "sonic_inference": (
                {
                    "topic": self.motion_token_topic,
                    "model_id": self.sonic_inferences[-1].model_id,
                    "model_sha256": self.sonic_inferences[-1].model_sha256,
                    "source": "same_run_online_post_fsq_decoder_input",
                }
                if self.sonic_inferences
                else None
            ),
        }

    def _start(self, task_id: str | None) -> None:
        now = time.monotonic()
        missing = self._ready(now)
        if missing:
            self.last_missing = missing
            self._set_error("recording start refused; missing: " + ", ".join(missing))
            return
        task = self._task(str(task_id or self.default_task_id))
        if task is None:
            self._set_error(f"unknown task_id={task_id!r}")
            return
        if not self.state_machine.start():
            return
        try:
            self.recorder.start_episode(
                {"task_id": task.task_id, "instruction_text": task.instruction_text},
                runtime_metadata=self._runtime_metadata(),
            )
        except Exception:
            self.state_machine.rearm()
            raise
        with self.lock:
            self.episode_anchor_floor = (
                self.images[self.active_rgb[0]][-1].sequence
                if self.images[self.active_rgb[0]]
                else -1
            )
            self.last_anchor_sequence = self.episode_anchor_floor
            self.episode_tick_count = 0
            self.last_recorded_inference_sequence = (
                self.sonic_inferences[-1].inference_sequence
                if self.sonic_inferences
                else -1
            )
            self.last_selected_camera_sequences = {
                name: self.images[name][-1].sequence if self.images[name] else -1
                for name in self.images
            }
            self.first_row_logged = False
            self.last_missing = []
        self.get_logger().info(
            f"[REC] recording started episode={self.recorder.episode_id:04d} "
            f"task={task.task_id}"
        )

    def _finish(self, source: str) -> None:
        if self.state_machine.phase != RecordingState.RECORDING:
            self._set_error(
                "finish refused: no active recording; after missing is empty, "
                "use left grip+A through the collection listener"
            )
            return
        if self.recorder.frame_count < self.min_frames:
            self._set_error(
                f"finish refused: {self.recorder.frame_count}/{self.min_frames} rows"
            )
            return
        try:
            result = self.recorder.end_episode(
                {"success": True, "reason": source}, self.min_frames
            )
        except Exception as exc:
            self._set_error(f"episode finalization failed: {exc}")
            return
        self.state_machine.finish()
        self.get_logger().info(
            f"[REC] saved frames={result['frames']} path={result['path']}"
        )

    def _abort(self, source: str) -> None:
        if not self.state_machine.abort():
            return
        result = self.recorder.discard_episode(source)
        self.get_logger().warning(
            f"[REC] discarded episode={result.get('episode_id')} "
            f"frames={result.get('frames')}"
        )

    def _control_tick(self) -> None:
        now = time.monotonic()
        with self.lock:
            events = tuple(self.manual_events)
            self.manual_events.clear()
        for event in events:
            command = str(event["command"])
            if command == "start":
                self._start(event.get("task_id"))
            elif command == "finish":
                self._finish("ros_control_finish")
            elif command == "abort":
                self._abort("ros_control_abort")
            elif command == "rearm":
                self.state_machine.rearm()

        # The manager already emits one-packet rising-edge booleans. Consume
        # every buffered packet once so a 20 ms pulse cannot be missed by timer
        # phase or executor scheduling.
        packets = self.pico.after(self.last_pico_control_packet_time)
        for packet in packets:
            self.last_pico_control_packet_time = max(
                self.last_pico_control_packet_time,
                packet.received_monotonic,
            )
            if not isinstance(packet, PicoSnapshot):
                continue
            if now - packet.received_monotonic > self.max_pico_delta_s:
                continue
            if packet.abort:
                self._abort("pico_abort")
                continue
            if not packet.toggle_collection:
                continue
            action = self.state_machine.toggle()
            if action == "start":
                self._start(None)
            elif action == "finish":
                self._finish("pico_toggle_finish")

    def _capture_tick(self) -> None:
        if not self.recorder.recording:
            return
        tick = time.monotonic()
        tick_sequence = self.episode_tick_count
        self.episode_tick_count += 1
        with self.lock:
            state = latest_at_or_before(self.states, tick, self.max_robot_delta_s)
            command = latest_at_or_before(
                self.commands, tick, self.max_robot_delta_s
            )
            sonic_inference = (
                matching_sonic_inference(
                    self.sonic_inferences,
                    command,
                    tick,
                    self.max_motion_token_delta_s,
                    self.motion_token_action_atol,
                )
                if command is not None
                else None
            )
            imu = latest_at_or_before(self.imus, tick, self.max_imu_delta_s)
            rgb_samples = {
                name: latest_at_or_before(
                    self.images[name], tick, self.max_camera_delta_s
                )
                for name in self.active_rgb
            }
            depth = (
                latest_at_or_before(
                    self.images[DEPTH_CAMERA_NAME], tick, self.max_camera_delta_s
                )
                if self.depth_enabled
                else None
            )
            status = (
                self.latest_state_machine
                if self.latest_state_machine is not None
                and self.latest_state_machine.received_monotonic <= tick
                else None
            )
            motion = (
                self.latest_motion_command
                if self.latest_motion_command is not None
                and self.latest_motion_command.received_monotonic <= tick
                else None
            )
            odom = (
                self.latest_odom
                if self.latest_odom is not None
                and self.latest_odom.received_monotonic <= tick
                else None
            )
            touch = (
                self.latest_touch
                if self.latest_touch is not None
                and self.latest_touch.received_monotonic <= tick
                else None
            )
            gripper_state = (
                latest_at_or_before(
                    self.gripper_states, tick, self.max_robot_delta_s
                )
                if self.gripper_enabled
                else None
            )
            gripper_command = (
                latest_at_or_before(
                    self.gripper_commands, tick, self.max_robot_delta_s
                )
                if self.gripper_enabled
                else None
            )
        pico = self.pico.latest_at_or_before(tick, self.max_pico_delta_s)
        smpl = self.smpl.latest_at_or_before(tick, self.max_smpl_delta_s)
        missing = []
        if state is None:
            missing.append("actuator_state")
        if command is None:
            missing.append("actuator_command")
        elif command.diagnostic_action_history is None:
            missing.append("diagnostic_action_history_10")
        if sonic_inference is None:
            missing.append("matched_sonic_inference_result")
        elif sonic_inference.inference_sequence <= self.last_recorded_inference_sequence:
            missing.append("new_sonic_inference_result")
        if imu is None:
            missing.append("imu")
        missing.extend(name for name, sample in rgb_samples.items() if sample is None)
        if self.depth_enabled and depth is None:
            missing.append("head_depth")
        if pico is None:
            missing.append("pico_raw")
        elif bool(self.requirements_cfg.get("require_pico_calibrated", True)) and (
            not isinstance(pico, PicoSnapshot)
            or not pico.calibrated
            or pico.stream_mode != 1
        ):
            missing.append("pico_calibrated_pose_mode")
        if smpl is None:
            missing.append("smpl_ref")
        elif not self._smpl_ready(smpl):
            missing.append("smpl_ref_ready")
        if status is None or tick - status.received_monotonic > self.max_state_machine_age_s:
            missing.append("state_machine_status")
        elif str(status.value.get("current", {}).get("name", "")) != self._required_state_name():
            missing.append("sonic_state")
        if self.gripper_enabled and gripper_state is None:
            missing.append("gripper_state")
        if self.gripper_enabled and gripper_command is None:
            missing.append("gripper_command")
        repeated_cameras = [
            name
            for name, sample in rgb_samples.items()
            if sample is not None
            and sample.sequence <= self.last_selected_camera_sequences[name]
        ]
        if repeated_cameras:
            if self.recorder.frame_count == 0:
                # Episode time starts at the first complete set of camera frames
                # received after the operator's start command.
                self.last_missing = [f"new_{name}_frame" for name in repeated_cameras]
                return
            missing.extend(f"repeated_{name}_frame" for name in repeated_cameras)
        if missing:
            self.last_missing = sorted(set(missing))
            self.dropped_control_ticks += 1
            if self.mode == "real" and bool(
                self.requirements_cfg.get("formal_collection", False)
            ):
                reason = "control_tick_sync_failure:" + ",".join(self.last_missing)
                self._set_error(reason)
                self._abort(reason)
            return
        assert state is not None and command is not None and imu is not None
        assert sonic_inference is not None
        assert command.diagnostic_action_history is not None
        assert pico is not None and smpl is not None and status is not None
        assert all(sample is not None for sample in rgb_samples.values())
        if self.depth_enabled:
            assert depth is not None
        if self.gripper_enabled:
            assert gripper_state is not None and gripper_command is not None
        try:
            trigger_intent = None
            if isinstance(pico, PicoSnapshot) and pico.triggers is not None:
                trigger_intent = list(pico.triggers)
            gravity = sonic_projected_gravity(
                imu.quaternion_wxyz, state.position[: len(BODY_JOINT_NAMES)]
            )
            self.recorder.capture(
                anchor_sequence=tick_sequence,
                observation={
                    "position": state.position,
                    "velocity": state.velocity,
                    "effort": state.effort,
                    "driver_temperature": state.driver_temperature,
                    "motor_temperature": state.motor_temperature,
                    "joint_present": state.joint_present,
                    "driver_temperature_present": state.driver_temperature_present,
                    "motor_temperature_present": state.motor_temperature_present,
                },
                action={
                    "position": command.position,
                    "velocity": command.velocity,
                    "torque": command.torque,
                    "kp": command.kp,
                    "kd": command.kd,
                    "joint_present": command.joint_present,
                },
                effective_continuous_action_29=command.diagnostic_action,
                diagnostic_action_history=command.diagnostic_action_history,
                motion_token=sonic_inference.motion_token,
                sonic_inference={
                    "sequence": sonic_inference.inference_sequence,
                    "model_id": sonic_inference.model_id,
                    "model_sha256": sonic_inference.model_sha256,
                    "policy_status": sonic_inference.policy_status,
                    "target_position": sonic_inference.target_position.tolist(),
                },
                projected_gravity=gravity,
                imu={
                    "orientation_wxyz": imu.quaternion_wxyz.tolist(),
                    "angular_velocity": imu.angular_velocity.tolist(),
                    "linear_acceleration": imu.linear_acceleration.tolist(),
                    "frame_id": imu.frame_id,
                },
                rgb_images={
                    name: sample.data
                    for name, sample in rgb_samples.items()
                    if sample is not None
                },
                depth_image=depth.data if depth is not None else None,
                depth_encoding=depth.encoding if depth is not None else None,
                pico=pico.json_safe_fields(),
                smpl_ref=smpl.json_safe_fields(),
                state_machine=status.value,
                motion_command=motion.value if motion is not None else None,
                auxiliary={
                    "control_tick_monotonic": tick,
                    "odom": odom.value if odom is not None else None,
                    "touch": touch.value if touch is not None else None,
                    "selected_camera_sequences": {
                        name: sample.sequence
                        for name, sample in rgb_samples.items()
                        if sample is not None
                    },
                },
                gripper={
                    "coordinate": ("motor_rad_diagnostic_only" if self.wrist_diagnostic
                                   else self.gripper_cfg["mapping"].get("coordinate_unit")),
                    "native_raw_sample": gripper_state.raw if gripper_state is not None else None,
                    "mapping": dict(self.gripper_cfg["mapping"]),
                    "trigger_intent": trigger_intent,
                    "state": (
                        gripper_state.values.tolist()
                        if gripper_state is not None
                        else None
                    ),
                    "state_is_measured": gripper_state is not None,
                    "command": (
                        gripper_command.values.tolist()
                        if gripper_command is not None
                        else None
                    ),
                    "command_is_final": gripper_command is not None,
                },
                source_timestamps={
                    "actuator_state": state.stamp_unix,
                    "actuator_command": command.stamp_unix,
                    "sonic_inference": sonic_inference.stamp_unix,
                    "imu": imu.stamp_unix,
                    **{
                        name: sample.stamp_unix
                        for name, sample in rgb_samples.items()
                        if sample is not None
                    },
                    **(
                        {"head_depth": depth.stamp_unix}
                        if depth is not None
                        else {}
                    ),
                    **(
                        {
                            "gripper_state": gripper_state.stamp_unix,
                            "gripper_command": gripper_command.stamp_unix,
                        }
                        if gripper_state is not None and gripper_command is not None
                        else {}
                    ),
                },
                source_receive_offsets={
                    "state_minus_tick": state.received_monotonic - tick,
                    "command_minus_tick": command.received_monotonic - tick,
                    "sonic_inference_minus_tick": (
                        sonic_inference.received_monotonic - tick
                    ),
                    "imu_minus_tick": imu.received_monotonic - tick,
                    "pico_minus_tick": pico.received_monotonic - tick,
                    "smpl_ref_minus_tick": smpl.received_monotonic - tick,
                    **(
                        {
                            "gripper_state_minus_tick": gripper_state.received_monotonic - tick,
                            "gripper_command_minus_tick": (
                                gripper_command.received_monotonic - tick
                            ),
                        }
                        if gripper_state is not None and gripper_command is not None
                        else {}
                    ),
                    **{
                        f"{name}_minus_tick": sample.received_monotonic - tick
                        for name, sample in rgb_samples.items()
                        if sample is not None
                    },
                },
            )
        except Exception as exc:
            self._set_error(f"capture rejected: {exc}")
            self.dropped_control_ticks += 1
            return
        self.last_recorded_inference_sequence = sonic_inference.inference_sequence
        self.last_anchor_sequence = tick_sequence
        for name, sample in rgb_samples.items():
            if sample is not None:
                self.last_selected_camera_sequences[name] = sample.sequence
        self.last_missing = []
        if not self.first_row_logged and self.recorder.frame_count > 0:
            self.get_logger().info("[REC] first synchronized row saved")
            self.first_row_logged = True

    def _status(self) -> dict[str, Any]:
        now = time.monotonic()
        with self.lock:
            counts = {
                "states": len(self.states),
                "commands": len(self.commands),
                "sonic_inferences": len(self.sonic_inferences),
                "imus": len(self.imus),
                "diagnostic_action_history": len(self.diagnostic_actions),
                **{f"{name}_images": len(self.images[name]) for name in self.images},
            }
            received_counts = {
                f"{name}_images": count + 1
                for name, count in self.camera_counts.items()
            }
        # In ARMED/FINISHED, readiness must be recomputed so a source restored
        # after a refused start clears automatically.  During RECORDING,
        # last_missing describes the most recent rejected synchronization tick.
        missing = (
            list(self.last_missing)
            if self.state_machine.phase == RecordingState.RECORDING
            else self._ready(now)
        )
        operator_hints: list[str] = []
        if "pico_raw" in missing and self.pico.received == 0:
            operator_hints.append(
                "PICO手柄按键在线不等于Body Tracking已发送：请在头显应用中开启"
                "Body Tracking，确认目标PC地址后点击Send；等待ABXY校准完成，松开"
                "全部按键，再重新按A+X。"
            )
        if "smpl_ref" in missing and "pico_raw" in missing:
            operator_hints.append(
                "smpl_ref是PICO body帧的下游结果；先恢复pico_raw，无需单独重启采集器。"
            )
        if (
            "sonic_inference_result" in missing
            and self.last_error is not None
            and "idle_reference" in self.last_error
        ):
            operator_hints.append(
                "SONIC仍在idle_reference；只有校准完成并进入live POSE后才会产生可采集推理结果。"
            )
        if self.state_machine.phase != RecordingState.RECORDING:
            operator_hints.append(
                "当前没有活动episode；accepted_rows保持0是预期行为。missing清空后按左握把+A开始；"
                "control.sh start仅作为备用。"
            )
        elif self.recorder.frame_count < self.min_frames:
            operator_hints.append(
                f"当前episode为{self.recorder.frame_count}/{self.min_frames}行；"
                "达到minimum_episode_rows前finish会被拒绝，但采集会继续。"
            )
        remaining_rows = max(0, self.min_frames - self.recorder.frame_count)
        return {
            "node": "elf3_real_collection",
            "wrist_diagnostic_only": self.wrist_diagnostic,
            "training_export_allowed": False if self.wrist_diagnostic else None,
            "mode": self.mode,
            "data_id": self.data_id,
            "phase": self.state_machine.phase,
            "episode_id": self.recorder.episode_id,
            "recording": self.recorder.recording,
            "accepted_rows": self.recorder.frame_count,
            "writer_saved_rows": (len(self.recorder.rows) if self.metadata_cfg.get('storage_backend') != 'nvidia_sonic_lerobot_v1' else None),
            "writer_buffered_rows": len(self.recorder.rows),
            "storage_backend": self.metadata_cfg.get('storage_backend', 'legacy_jpeg'),
            "minimum_episode_rows": self.min_frames,
            "remaining_rows_to_finish": remaining_rows,
            "active_episode_path": (
                str(self.recorder.inprogress_dir)
                if self.recorder.inprogress_dir is not None
                else None
            ),
            "writer_queue": self.recorder.writer_queue_size(),
            "dropped_control_ticks": self.dropped_control_ticks,
            "missing": missing,
            "last_error": self.last_error,
            "pico_error": self.pico.error,
            "smpl_ref_error": self.smpl.error,
            "pico_packets_received": self.pico.received,
            "smpl_ref_packets_received": self.smpl.received,
            "operator_hints": operator_hints,
            "counts": counts,
            "received_counts": received_counts,
            "source_root": str(self.source_root),
        }

    def _publish_status(self) -> None:
        message = String()
        status = self._status()
        message.data = json.dumps(status, ensure_ascii=False, allow_nan=False)
        self.status_publisher.publish(message)
        if self.wrist_diagnostic and time.monotonic()-getattr(self, '_diagnostic_print_at', 0) >= 5:
            self._diagnostic_print_at = time.monotonic()
            self.get_logger().info('[THREE RGB REC] ' + json.dumps({
                key: status[key] for key in ('phase', 'accepted_rows', 'writer_saved_rows',
                    'dropped_control_ticks', 'missing', 'received_counts')}, ensure_ascii=False))

    def _set_error(self, text: str) -> None:
        if text != self.last_error:
            self.get_logger().error(text)
        self.last_error = text

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.pico.close()
        self.smpl.close()
        if self.recorder.recording:
            try:
                if self.recorder.frame_count > 0:
                    result = self.recorder.preserve_interrupted_episode(
                        "process_shutdown", min_frames=self.min_frames
                    )
                    self.get_logger().warning(
                        "[REC] shutdown preserved incomplete episode "
                        f"frames={result.get('saved_frames')} path={result.get('path')}"
                    )
                else:
                    self.recorder.discard_episode("empty_process_shutdown")
            except Exception as exc:
                self.get_logger().error(f"could not preserve shutdown episode: {exc}")


def _arguments(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ELF3 synchronized LeRobot source recorder")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--data-id", default=os.environ.get("DATA_ID"))
    arguments, _ = parser.parse_known_args(argv)
    if arguments.data_id is not None and not str(arguments.data_id).strip():
        arguments.data_id = None
    return arguments


def main(argv: list[str] | None = None) -> int:
    arguments = _arguments(sys.argv[1:] if argv is None else argv)
    rclpy.init(args=None)
    node: RealCollectionNode | None = None
    executor: MultiThreadedExecutor | None = None
    try:
        node = RealCollectionNode(arguments.config, arguments.data_id)
        executor = MultiThreadedExecutor(num_threads=6)
        executor.add_node(node)
        executor.spin()
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        print(f"fatal: {exc}", file=sys.stderr)
        return 1
    finally:
        if executor is not None:
            executor.shutdown(timeout_sec=2.0)
        if node is not None:
            node.close()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
