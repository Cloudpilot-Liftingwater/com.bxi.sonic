#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import yaml

from real_robot_collection.contracts import (
    BODY_JOINT_NAMES,
    GRIPPER_COMMAND_NAMES,
    GRIPPER_STATE_JOINT_NAMES,
    PUBLISHED_FPS,
    tasks_from_config,
)
from real_robot_collection.recorder import DEPTH_CAMERA_NAME, RGB_CAMERA_NAMES
from real_robot_collection.native66 import validate_mapping
from real_robot_collection.wrist_diagnostic import enabled as wrist_diagnostic_enabled


SECTIONS = (
    "dataset",
    "robot",
    "sequence",
    "gripper",
    "motion_token",
    "pico",
    "smpl_ref",
    "cameras",
    "requirements",
    "metadata",
)


def placeholder_paths(value, prefix: str = "") -> list[str]:
    if isinstance(value, dict):
        result = []
        for key, nested in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            result.extend(placeholder_paths(nested, path))
        return result
    if isinstance(value, list):
        result = []
        for index, nested in enumerate(value):
            result.extend(placeholder_paths(nested, f"{prefix}[{index}]"))
        return result
    text = str(value)
    return [prefix] if "REPLACE_" in text else []


def _positive(mapping: dict, key: str) -> float:
    value = float(mapping.get(key, 0.0))
    if value <= 0.0:
        raise ValueError(f"{key} must be positive")
    return value


def validate_config(config: dict, *, formal: bool, allow_placeholders: bool) -> None:
    if not isinstance(config, dict):
        raise ValueError("root is not a mapping")
    mode = str(config.get("mode", ""))
    if mode not in ("sim", "hybrid_debug", "real"):
        raise ValueError("mode must be sim, hybrid_debug, or real")
    for section in SECTIONS:
        if not isinstance(config.get(section), dict):
            raise ValueError(f"missing mapping section {section}")
    dataset = config["dataset"]
    robot = config["robot"]
    sequence = config["sequence"]
    cameras = config["cameras"]
    requirements = config["requirements"]
    gripper = config["gripper"]
    wrist_diagnostic = wrist_diagnostic_enabled(config)
    if wrist_diagnostic and formal:
        raise ValueError("wrist diagnostic cannot be accepted as formal collection")
    if not wrist_diagnostic and dataset.get("embodiment") != "elf3_native_1d":
        raise ValueError("native66 config required; legacy config is not silently converted")
    if not wrist_diagnostic:
        validate_mapping(gripper.get("mapping"))
    motion_token = config["motion_token"]
    tasks = tasks_from_config(dataset)
    default_task = str(dataset.get("default_task_id", ""))
    if default_task not in {task.task_id for task in tasks}:
        raise ValueError("dataset.default_task_id is not present in dataset.tasks")
    expected_fps = (50 if wrist_diagnostic and config.get("metadata", {}).get(
        "capture_profile") == "sonic50_latest_rgb" else PUBLISHED_FPS)
    if int(_positive(dataset, "fps")) != expected_fps:
        raise ValueError(f"dataset.fps must be {expected_fps}")
    _positive(dataset, "min_episode_frames")
    _positive(dataset, "queue_capacity")
    if tuple(robot.get("body_joint_names", ())) != BODY_JOINT_NAMES:
        raise ValueError("robot.body_joint_names is not canonical 29D SONIC order")
    for key in (
        "actuator_state_topic",
        "actuator_command_topic",
        "imu_topic",
        "state_machine_topic",
        "motion_commands_topic",
    ):
        if not str(robot.get(key, "")).startswith("/"):
            raise ValueError(f"robot.{key} must be an absolute ROS topic")
    if not str(motion_token.get("topic", "")).startswith("/"):
        raise ValueError("motion_token.topic must be an absolute ROS topic")
    _positive(motion_token, "max_delta_ms")
    _positive(motion_token, "action_match_atol")
    expected_prefix = "/hardware/" if mode == "real" else "/simulation/"
    for key in (
        "actuator_state_topic",
        "actuator_command_topic",
        "imu_topic",
        "state_machine_topic",
    ):
        if not str(robot[key]).startswith(expected_prefix):
            raise ValueError(
                f"mode={mode} requires robot.{key} under {expected_prefix}"
            )
    if sequence.get("mode") not in SequenceModes:
        raise ValueError(f"sequence.mode must be one of {sorted(SequenceModes)}")
    if int(sequence.get("history_size", 0)) < 10:
        raise ValueError("sequence.history_size must be at least 10")
    for section, defaults in (
        (config["pico"], ("host", "port", "topic", "max_delta_ms")),
        (config["smpl_ref"], ("host", "port", "topic", "max_delta_ms")),
    ):
        for key in defaults:
            if key not in section or str(section[key]).strip() == "":
                raise ValueError(f"packed stream missing {key}")
    active_rgb = []
    topics = []
    for name in (*RGB_CAMERA_NAMES, DEPTH_CAMERA_NAME):
        definition = cameras.get(name)
        if not isinstance(definition, dict):
            raise ValueError(f"missing cameras.{name}")
        enabled = bool(definition.get("enabled", True))
        if not enabled:
            continue
        if name in RGB_CAMERA_NAMES:
            active_rgb.append(name)
            if definition.get("message_type") not in ("raw", "compressed"):
                raise ValueError(f"invalid cameras.{name}.message_type")
            for key, expected in (
                ("width", 1920 if wrist_diagnostic and name == 'head_rgb' else 640),
                ("height", 1080 if wrist_diagnostic and name == 'head_rgb' else 480),
                ("fps", PUBLISHED_FPS),
            ):
                if int(definition.get(key, -1)) != expected:
                    raise ValueError(f"cameras.{name}.{key} must be {expected}")
        elif definition.get("message_type") != "raw":
            raise ValueError("head_depth.message_type must be raw")
        for key in ("topic", "camera_info_topic"):
            topic = str(definition.get(key, ""))
            if not topic.startswith("/"):
                raise ValueError(f"cameras.{name}.{key} must be an absolute topic")
            topics.append(topic)
    if "head_rgb" not in active_rgb and not wrist_diagnostic:
        raise ValueError("head_rgb must be enabled")
    if len(set(topics)) != len(topics):
        raise ValueError("enabled camera topics must be distinct")
    declared_formal = bool(requirements.get("formal_collection", False))
    if formal or declared_formal:
        if mode != "real":
            raise ValueError("formal collection requires mode=real")
        if tuple(active_rgb) != RGB_CAMERA_NAMES:
            raise ValueError("formal collection requires all three RGB cameras")
        if not bool(cameras[DEPTH_CAMERA_NAME].get("enabled", True)):
            raise ValueError("formal collection requires RealSense head depth")
        depth = cameras[DEPTH_CAMERA_NAME]
        if depth.get("aligned_to") != "head_rgb":
            raise ValueError("formal collection requires head depth aligned to head_rgb")
        if "/aligned_depth_to_color/" not in str(depth.get("topic", "")):
            raise ValueError("formal collection requires the aligned depth image topic")
        if not bool(gripper.get("enabled", False)):
            raise ValueError("formal collection requires gripper feedback and command")
        if tuple(gripper.get("state_joint_names", ())) != GRIPPER_STATE_JOINT_NAMES:
            raise ValueError("gripper.state_joint_names must use the fixed two-joint order")
        if tuple(gripper.get("command_joint_names", ())) != GRIPPER_COMMAND_NAMES:
            raise ValueError(
                "gripper.command_joint_names must use the fixed left/right order"
            )
        for key in ("sample_topic",):
            if not str(gripper.get(key, "")).startswith("/"):
                raise ValueError(f"gripper.{key} must be an absolute ROS topic")
        for name in RGB_CAMERA_NAMES:
            definition = cameras[name]
            if not str(definition.get("serial", "")).strip():
                raise ValueError(f"cameras.{name}.serial is required")
        if motion_token.get("source") != (
            "customer_sonic_step28800_post_fsq_decoder_input"
        ):
            raise ValueError("motion_token.source does not match the ELF3 baseline")
        if int(motion_token.get("dimension", -1)) != 64:
            raise ValueError("motion_token.dimension must be 64")
        if int(motion_token.get("fsq_levels", -1)) != 32:
            raise ValueError("motion_token.fsq_levels must be 32")
        if motion_token.get("model_id") != "elf3_sonic_step28800_action_post_fsq":
            raise ValueError("motion_token.model_id does not match the deployed model")
        expected_sha256 = (
            "50303783f72ca61674b54987cab20abd11"
            "d0ced25c7042062fb03ef6355cdfdf"
        )
        if motion_token.get("sha256") != expected_sha256:
            raise ValueError("motion_token.sha256 does not match the verified model")
        if motion_token.get("required_policy_status") != "live_reference":
            raise ValueError("formal collection requires live_reference Sonic output")
    placeholders = placeholder_paths(config)
    if placeholders and not allow_placeholders:
        raise ValueError(
            "replace deployment placeholders: " + ", ".join(placeholders)
        )


SequenceModes = {"frame_id", "counter", "frame_id_or_counter"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate ELF3 collection configuration")
    parser.add_argument("config", type=Path)
    parser.add_argument("--allow-placeholders", action="store_true")
    parser.add_argument("--formal", action="store_true")
    args = parser.parse_args()
    try:
        with args.config.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        validate_config(
            config,
            formal=bool(args.formal),
            allow_placeholders=bool(args.allow_placeholders),
        )
    except Exception as exc:
        print(f"CONFIG ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"CONFIG OK: {args.config} mode={config['mode']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
