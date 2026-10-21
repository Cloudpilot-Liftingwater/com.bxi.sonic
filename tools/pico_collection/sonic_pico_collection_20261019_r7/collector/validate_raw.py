#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from real_robot_collection.native66 import validate_gripper_row

from real_robot_collection.contracts import (
    DIAGNOSTIC_ACTION_DIM,
    DIAGNOSTIC_ACTION_HISTORY,
    BODY_JOINT_NAMES,
    CANONICAL_JOINT_NAMES,
    PUBLISHED_FPS,
    STATE_DIM,
    WBC_ACTION_DIM,
    validate_motion_token,
)
from real_robot_collection.pico import RAW_REQUIRED_FIELDS, SMPL_REQUIRED_FIELDS
from real_robot_collection.recorder import (
    DEPTH_CAMERA_NAME,
    RGB_CAMERA_NAMES,
    SCHEMA_VERSION,
)


class ValidationError(RuntimeError):
    pass


def read_json(path: Path):
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception as exc:
        raise ValidationError(f"cannot read {path}: {exc}") from exc


def read_rows(path: Path) -> list[dict]:
    rows = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, text in enumerate(handle, 1):
                if text.strip():
                    rows.append(json.loads(text))
    except Exception as exc:
        raise ValidationError(f"cannot read {path}: {exc}") from exc
    return rows


def _finite(row: dict, key: str, shape: tuple[int, ...], episode: Path, index: int):
    value = np.asarray(row.get(key), dtype=np.float64)
    if value.shape != shape or not np.isfinite(value).all():
        raise ValidationError(
            f"{episode}: row {index} invalid {key}; expected {shape}, got {value.shape}"
        )
    return value


def _mask(row: dict, key: str, size: int, episode: Path, index: int):
    value = np.asarray(row.get(key), dtype=bool).reshape(-1)
    if value.shape != (size,):
        raise ValidationError(f"{episode}: row {index} invalid {key}")
    return value


def _contains_placeholder(value) -> bool:
    if isinstance(value, dict):
        return any(_contains_placeholder(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_placeholder(item) for item in value)
    return "REPLACE_" in str(value)


def validate_episode(path: Path, decode_all_images: bool, formal: bool = False) -> dict:
    meta = read_json(path / "meta.json")
    rows = read_rows(path / "frames.jsonl")
    if meta.get("schema_version") != SCHEMA_VERSION:
        raise ValidationError(f"{path}: unsupported schema {meta.get('schema_version')!r}")
    if not rows:
        raise ValidationError(f"{path}: no rows")
    if formal and len(rows) < 100:
        raise ValidationError(f"{path}: formal episode must have at least 100 rows")
    if int(meta.get("num_frames", -1)) != len(rows):
        raise ValidationError(f"{path}: meta num_frames does not match rows")
    termination = meta.get("termination")
    if not isinstance(termination, dict) or termination.get("success") is not True:
        raise ValidationError(f"{path}: unsuccessful episode {termination!r}")
    joint_contract = meta.get("joint_contract", {})
    if tuple(joint_contract.get("canonical_names", ())) != CANONICAL_JOINT_NAMES:
        raise ValidationError(f"{path}: canonical joint names/order mismatch")
    if int(meta.get("fps", -1)) != PUBLISHED_FPS:
        raise ValidationError(f"{path}: fps must be {PUBLISHED_FPS}")
    token_contract = meta.get("diagnostic_action_contract", {})
    if int(token_contract.get("dimension", -1)) != DIAGNOSTIC_ACTION_DIM:
        raise ValidationError(f"{path}: diagnostic action dimension mismatch")
    if token_contract.get("source") != "reconstructed_from_final_ActuatorCmds.pos":
        raise ValidationError(f"{path}: unexpected diagnostic action source")
    if token_contract.get("is_training_motion_token") is not False:
        raise ValidationError(f"{path}: 29-D diagnostic is mislabeled as training token")
    published_contract = meta.get("published_dataset_contract", {})
    if published_contract.get("motion_token_is_offline_sidecar") is not False:
        raise ValidationError(f"{path}: motion token is not declared as online")
    if published_contract.get("motion_token_is_online_same_inference_output") is not True:
        raise ValidationError(f"{path}: motion token lacks same-inference provenance")
    camera_contract = meta.get("camera_contract", {})
    rgb_names = tuple(camera_contract.get("rgb", ()))
    if not rgb_names or any(name not in RGB_CAMERA_NAMES for name in rgb_names):
        raise ValidationError(f"{path}: invalid RGB camera contract {rgb_names}")
    depth_name = camera_contract.get("depth")
    if depth_name not in (None, DEPTH_CAMERA_NAME):
        raise ValidationError(f"{path}: invalid depth contract {depth_name!r}")
    if formal:
        if meta.get("mode") != "real":
            raise ValidationError(f"{path}: formal validation requires mode=real")
        if rgb_names != RGB_CAMERA_NAMES or depth_name != DEPTH_CAMERA_NAME:
            raise ValidationError(f"{path}: formal data requires 3 RGB + head depth")
        if _contains_placeholder(meta):
            raise ValidationError(f"{path}: formal metadata contains REPLACE_ placeholders")

    anchors = []
    previous_frame_indices = {name: None for name in ("pico.raw", "pico.smpl_ref")}
    previous_camera_sequences = {name: None for name in RGB_CAMERA_NAMES}
    previous_inference_sequence = -1
    sonic_model_sha256 = None
    for expected_index, row in enumerate(rows):
        if int(row.get("idx", -1)) != expected_index:
            raise ValidationError(f"{path}: noncontiguous idx at row {expected_index}")
        anchors.append(int(row.get("anchor_sequence", -1)))
        for key in (
            "observation.state",
            "observation.velocity",
            "observation.effort",
            "observation.driver_temperature",
            "observation.motor_temperature",
            "action",
            "action.velocity",
            "action.torque",
            "action.kp",
            "action.kd",
        ):
            _finite(row, key, (len(CANONICAL_JOINT_NAMES),), path, expected_index)
        observation_mask = _mask(
            row,
            "observation.joint_present",
            len(CANONICAL_JOINT_NAMES),
            path,
            expected_index,
        )
        action_mask = _mask(
            row,
            "action.joint_present",
            len(CANONICAL_JOINT_NAMES),
            path,
            expected_index,
        )
        if not observation_mask[: len(BODY_JOINT_NAMES)].all():
            raise ValidationError(f"{path}: row {expected_index} missing body state joints")
        if not action_mask[: len(BODY_JOINT_NAMES)].all():
            raise ValidationError(f"{path}: row {expected_index} missing body action joints")
        gravity = _finite(
            row, "observation.projected_gravity", (3,), path, expected_index
        )
        if not np.isclose(np.linalg.norm(gravity), 1.0, atol=1.0e-4, rtol=0.0):
            raise ValidationError(f"{path}: row {expected_index} gravity is not unit")
        _finite(
            row,
            "sonic.effective_continuous_action_29",
            (DIAGNOSTIC_ACTION_DIM,),
            path,
            expected_index,
        )
        _finite(
            row,
            "sonic.effective_continuous_action_history_10x29",
            (DIAGNOSTIC_ACTION_HISTORY, DIAGNOSTIC_ACTION_DIM),
            path,
            expected_index,
        )
        try:
            motion_token = validate_motion_token(row.get("action.motion_token"))
        except (TypeError, ValueError) as exc:
            raise ValidationError(
                f"{path}: row {expected_index} invalid online motion token: {exc}"
            ) from exc
        inference = row.get("sonic.inference")
        if not isinstance(inference, dict):
            raise ValidationError(f"{path}: row {expected_index} missing Sonic inference")
        inference_sequence = int(inference.get("sequence", -1))
        if inference_sequence <= previous_inference_sequence:
            raise ValidationError(
                f"{path}: row {expected_index} Sonic inference sequence is not increasing"
            )
        previous_inference_sequence = inference_sequence
        if inference.get("source") != "same_run_online_post_fsq_decoder_input":
            raise ValidationError(f"{path}: row {expected_index} wrong token source")
        if formal and inference.get("policy_status") != "live_reference":
            raise ValidationError(f"{path}: row {expected_index} token is not from live Sonic")
        model_sha256 = str(inference.get("model_sha256", "")).lower()
        if len(model_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in model_sha256
        ):
            raise ValidationError(f"{path}: row {expected_index} invalid Sonic checksum")
        if sonic_model_sha256 is None:
            sonic_model_sha256 = model_sha256
        elif model_sha256 != sonic_model_sha256:
            raise ValidationError(f"{path}: Sonic model changed within the episode")
        policy_target = np.asarray(inference.get("target_position"), dtype=np.float64)
        action_target = np.asarray(row.get("action"), dtype=np.float64)
        if policy_target.shape != (len(CANONICAL_JOINT_NAMES),) or not np.allclose(
            policy_target[: len(BODY_JOINT_NAMES)],
            action_target[: len(BODY_JOINT_NAMES)],
            rtol=0.0,
            atol=1.0e-5,
        ):
            raise ValidationError(
                f"{path}: row {expected_index} policy target does not match motor command"
            )
        _ = motion_token
        imu = row.get("observation.imu")
        if not isinstance(imu, dict):
            raise ValidationError(f"{path}: row {expected_index} missing IMU")
        for key, size in (
            ("orientation_wxyz", 4),
            ("angular_velocity", 3),
            ("linear_acceleration", 3),
        ):
            value = np.asarray(imu.get(key), dtype=np.float64).reshape(-1)
            if value.shape != (size,) or not np.isfinite(value).all():
                raise ValidationError(f"{path}: row {expected_index} invalid IMU {key}")
        state_machine = row.get("state_machine", {})
        current_name = state_machine.get("current", {}).get("name")
        if current_name != "com.bxi.sonic/sonic_teleop":
            raise ValidationError(
                f"{path}: row {expected_index} was not captured in SONIC state"
            )
        for key, required in (
            ("pico.raw", RAW_REQUIRED_FIELDS),
            ("pico.smpl_ref", SMPL_REQUIRED_FIELDS),
        ):
            packet = row.get(key)
            if not isinstance(packet, dict):
                raise ValidationError(f"{path}: row {expected_index} missing {key}")
            missing = [name for name in required if name not in packet]
            if missing:
                raise ValidationError(
                    f"{path}: row {expected_index} {key} missing {missing}"
                )
            indices = np.asarray(packet["frame_index"], dtype=np.int64).reshape(-1)
            if indices.size == 0 or np.any(np.diff(indices) < 0):
                raise ValidationError(f"{path}: row {expected_index} invalid {key} frame_index")
            previous = previous_frame_indices[key]
            if previous is not None and int(indices[-1]) < previous:
                raise ValidationError(f"{path}: row {expected_index} {key} moved backwards")
            previous_frame_indices[key] = int(indices[-1])
        raw_packet = row["pico.raw"]
        if not bool(np.asarray(raw_packet["calibration_ready"]).reshape(-1)[-1]):
            raise ValidationError(f"{path}: row {expected_index} PICO is not calibrated")
        if int(np.asarray(raw_packet["stream_mode"]).reshape(-1)[-1]) != 1:
            raise ValidationError(f"{path}: row {expected_index} PICO is not in POSE mode")
        smpl_packet = row["pico.smpl_ref"]
        if not bool(np.asarray(smpl_packet["source_ready"]).reshape(-1)[-1]):
            raise ValidationError(f"{path}: row {expected_index} smpl_ref is not ready")
        if int(np.asarray(smpl_packet["source_stream_mode"]).reshape(-1)[-1]) != 1:
            raise ValidationError(f"{path}: row {expected_index} smpl_ref mode is invalid")
        if not bool(
            np.asarray(smpl_packet["source_calibration_ready"]).reshape(-1)[-1]
        ):
            raise ValidationError(f"{path}: row {expected_index} smpl_ref is uncalibrated")
        images = row.get("observation.images")
        if not isinstance(images, dict) or tuple(images) != rgb_names:
            raise ValidationError(f"{path}: row {expected_index} RGB path contract mismatch")
        for name in rgb_names:
            image_path = path / str(images[name])
            if not image_path.is_file() or image_path.stat().st_size <= 0:
                raise ValidationError(f"{path}: missing {images[name]}")
            if decode_all_images or expected_index in (0, len(rows) - 1):
                image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if image is None or image.ndim != 3 or image.shape[2] != 3:
                    raise ValidationError(f"{path}: undecodable RGB image {images[name]}")
                if formal and image.shape != (480, 640, 3):
                    raise ValidationError(
                        f"{path}: {images[name]} must be 640x480, got {image.shape}"
                    )
        depth = row.get("observation.depth")
        if depth_name is None:
            if depth is not None:
                raise ValidationError(f"{path}: row {expected_index} unexpected depth")
        else:
            if not isinstance(depth, dict):
                raise ValidationError(f"{path}: row {expected_index} missing depth")
            depth_path = path / str(depth.get("path", ""))
            try:
                array = np.load(depth_path, allow_pickle=False, mmap_mode="r")
            except Exception as exc:
                raise ValidationError(f"{path}: cannot load depth {depth_path}: {exc}") from exc
            if tuple(array.shape) != tuple(depth.get("shape", ())) or str(array.dtype) != str(
                depth.get("dtype")
            ):
                raise ValidationError(f"{path}: row {expected_index} depth metadata mismatch")
        for dictionary_key in (
            "source_timestamps_unix",
            "source_receive_offsets_s",
        ):
            values = row.get(dictionary_key)
            if not isinstance(values, dict) or not values:
                raise ValidationError(f"{path}: row {expected_index} missing {dictionary_key}")
            if not np.isfinite(np.asarray(list(values.values()), dtype=np.float64)).all():
                raise ValidationError(f"{path}: row {expected_index} invalid {dictionary_key}")

        if formal:
            state_33 = _finite(
                row, "dataset.observation.state", (STATE_DIM,), path, expected_index
            )
            action_33 = _finite(
                row, "dataset.action.wbc", (WBC_ACTION_DIM,), path, expected_index
            )
            gripper = row.get("gripper")
            if not isinstance(gripper, dict):
                raise ValidationError(f"{path}: row {expected_index} missing gripper")
            try:
                validate_gripper_row(gripper)
            except ValueError as exc:
                raise ValidationError(f"{path}: row {expected_index}: {exc}") from exc
            if not np.allclose(state_33[29:], gripper["state"], rtol=0, atol=1e-7) or not np.allclose(action_33[29:], gripper["command"], rtol=0, atol=1e-7):
                raise ValidationError(f"{path}: row {expected_index} native gripper vector mismatch")
            if gripper.get("state_is_measured") is not True:
                raise ValidationError(f"{path}: row {expected_index} gripper state is not measured")
            if gripper.get("command_is_final") is not True:
                raise ValidationError(f"{path}: row {expected_index} gripper command is not final")
            _ = state_33
            offsets = row["source_receive_offsets_s"]
            future = {key: value for key, value in offsets.items() if value > 1.0e-9}
            if future:
                raise ValidationError(
                    f"{path}: row {expected_index} selected future samples {future}"
                )
            sequences = row.get("auxiliary", {}).get("selected_camera_sequences", {})
            for name in RGB_CAMERA_NAMES:
                sequence = int(sequences.get(name, -1))
                previous = previous_camera_sequences[name]
                if sequence < 0 or (previous is not None and sequence <= previous):
                    raise ValidationError(
                        f"{path}: row {expected_index} repeated/missing {name} frame"
                    )
                previous_camera_sequences[name] = sequence

    anchors_np = np.asarray(anchors, dtype=np.int64)
    if np.any(anchors_np < 0) or np.any(np.diff(anchors_np) <= 0):
        raise ValidationError(f"{path}: anchor_sequence is not strictly increasing")
    return {
        "episode": path.name,
        "frames": len(rows),
        "task": meta.get("task_name"),
        "mode": meta.get("mode"),
    }


def source_path(value: Path) -> Path:
    value = value.expanduser().resolve()
    if (value / "lerobot_source").is_dir():
        return value / "lerobot_source"
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate ELF3 LeRobot source episodes")
    parser.add_argument("source", type=Path)
    parser.add_argument("--decode-all-images", action="store_true")
    parser.add_argument("--formal", action="store_true")
    args = parser.parse_args()
    root = source_path(args.source)
    if not root.is_dir():
        print(f"ERROR: source directory does not exist: {root}", file=sys.stderr)
        return 1
    incomplete = sorted(root.glob(".episode_*.inprogress"))
    if incomplete:
        print("ERROR: incomplete episode directories exist:", file=sys.stderr)
        for path in incomplete:
            print(f"  {path}", file=sys.stderr)
        return 1
    episodes = sorted(path for path in root.glob("episode_*") if path.is_dir())
    if not episodes:
        print(f"ERROR: no episodes in {root}", file=sys.stderr)
        return 1
    try:
        reports = [
            validate_episode(path, args.decode_all_images, formal=args.formal)
            for path in episodes
        ]
    except ValidationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    for report in reports:
        print(
            f"OK {report['episode']}: frames={report['frames']} "
            f"mode={report['mode']} task={report['task']!r}"
        )
    print(f"PASS: {len(reports)} episode(s) validated in {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
