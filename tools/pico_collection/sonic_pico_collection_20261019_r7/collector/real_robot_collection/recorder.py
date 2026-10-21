from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
from pathlib import Path
from queue import Empty, Full, Queue
import shutil
import threading
import time
from typing import Any

import cv2
import numpy as np
from .native66 import validate_gripper_row
from .wrist_diagnostic import SCHEMA as WRIST_SCHEMA, WRISTS, validate_raw_row
from .gripper_identity import validate_identities

from .contracts import (
    DIAGNOSTIC_ACTION_DIM,
    DIAGNOSTIC_ACTION_HISTORY,
    BODY_JOINT_NAMES,
    CANONICAL_JOINT_NAMES,
    GRIPPER_COMMAND_NAMES,
    GRIPPER_STATE_JOINT_NAMES,
    MOTION_TOKEN_DIM,
    PUBLISHED_FPS,
    finite_vector,
    pack_action_31,
    pack_state_31,
    validate_effective_continuous_action_29,
    validate_motion_token,
)


RGB_CAMERA_NAMES = ("head_rgb", "left_wrist_rgb", "right_wrist_rgb")
CAMERA_NAMES = RGB_CAMERA_NAMES  # compatibility for external imports
DEPTH_CAMERA_NAME = "head_depth"
SCHEMA_VERSION = "elf3_native66_source/v1"


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256_file(path: Path) -> str | None:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


class RealEpisodeRecorder:
    def __init__(
        self,
        source_root: Path,
        *,
        mode: str,
        fps: int,
        jpeg_quality: int,
        queue_capacity: int,
        rgb_camera_names: tuple[str, ...],
        depth_enabled: bool,
        camera_metadata: dict[str, Any],
        common_metadata: dict[str, Any],
        config_path: Path,
        wrist_diagnostic: bool = False,
    ) -> None:
        self.source_root = Path(source_root)
        self.mode = str(mode)
        self.fps = int(fps)
        expected_fps = (50 if wrist_diagnostic and
                        common_metadata.get('capture_profile') == 'sonic50_latest_rgb' else PUBLISHED_FPS)
        if self.fps != expected_fps:
            raise ValueError(
                f"dataset fps must be {expected_fps} for the published data contract"
            )
        self.jpeg_quality = int(jpeg_quality)
        self.rgb_camera_names = tuple(rgb_camera_names)
        self.depth_enabled = bool(depth_enabled)
        self.wrist_diagnostic = bool(wrist_diagnostic)
        if self.wrist_diagnostic and (self.mode != 'real' or self.rgb_camera_names != WRISTS or self.depth_enabled):
            raise ValueError('wrist diagnostic recorder requires real dual-wrist-only input')
        if not self.rgb_camera_names:
            raise ValueError("at least one RGB camera must be enabled")
        if any(name not in RGB_CAMERA_NAMES for name in self.rgb_camera_names):
            raise ValueError(f"unsupported RGB camera set {self.rgb_camera_names}")
        self.camera_metadata = camera_metadata
        self.common_metadata = common_metadata
        self.gripper_identities = (validate_identities(common_metadata.get('gripper_identities'))
                                   if self.wrist_diagnostic else None)
        self.config_path = Path(config_path)
        self.queue: Queue = Queue(maxsize=int(queue_capacity))
        self.episode_id = self._next_episode_id()
        self.recording = False
        self.frame_count = 0
        self.rows: list[dict] = []
        self.errors: list[str] = []
        self.worker: threading.Thread | None = None
        self.stop_worker = False
        self.discarding = False
        self.inprogress_dir: Path | None = None
        self.final_dir: Path | None = None
        self.started_unix = 0.0
        self.last_anchor_sequence = -1
        self.last_sonic_inference_sequence = -1
        self.meta: dict[str, Any] = {}

    def _next_episode_id(self) -> int:
        self.source_root.mkdir(parents=True, exist_ok=True)
        ids = []
        for path in self.source_root.glob("episode_*"):
            suffix = path.name.rsplit("_", 1)[-1]
            if suffix.isdigit():
                ids.append(int(suffix))
        return max(ids, default=-1) + 1

    def start_episode(
        self,
        task: dict[str, Any],
        *,
        runtime_metadata: dict[str, Any],
    ) -> int:
        if self.recording:
            raise RuntimeError("cannot start a second episode while recording")
        episode_name = f"episode_{self.episode_id:04d}"
        self.inprogress_dir = self.source_root / f".{episode_name}.inprogress"
        self.final_dir = self.source_root / episode_name
        if self.inprogress_dir.exists() or self.final_dir.exists():
            raise FileExistsError(f"episode path already exists: {episode_name}")
        for camera in self.rgb_camera_names:
            (self.inprogress_dir / camera / "rgb").mkdir(parents=True, exist_ok=True)
        if self.depth_enabled:
            (self.inprogress_dir / DEPTH_CAMERA_NAME / "depth").mkdir(
                parents=True, exist_ok=True
            )
        self.recording = True
        self.frame_count = 0
        self.rows = []
        self.errors = []
        self.stop_worker = False
        self.discarding = False
        self.started_unix = time.time()
        self.first_source_stamp = None
        self.last_source_stamp = None
        self.last_anchor_sequence = -1
        self.last_sonic_inference_sequence = -1
        nested = dict(self.common_metadata)
        nested.update(task)
        nested.update(runtime_metadata)
        self.meta = {
            "schema_version": WRIST_SCHEMA if self.wrist_diagnostic else SCHEMA_VERSION,
            "episode_id": self.episode_id,
            "mode": self.mode,
            "task_name": str(task["instruction_text"]),
            "task_id": str(task["task_id"]),
            "fps": self.fps,
            "start_time_utc": utc_now(),
            "start_time_unix": self.started_unix,
            "metadata": nested,
            "configuration": {
                "path": str(self.config_path.resolve()),
                "sha256": sha256_file(self.config_path),
            },
            "joint_contract": {
                "canonical_names": list(CANONICAL_JOINT_NAMES),
                "dimension": len(CANONICAL_JOINT_NAMES),
                "missing_value": 0.0,
                "missing_value_is_valid_only_when_presence_mask_is_false": True,
            },
            "diagnostic_action_contract": {
                "name": "sonic_effective_continuous_action_29",
                "dimension": DIAGNOSTIC_ACTION_DIM,
                "history": DIAGNOSTIC_ACTION_HISTORY,
                "source": "reconstructed_from_final_ActuatorCmds.pos",
                "is_training_motion_token": False,
            },
            "published_dataset_contract": {
                "codebase_version": "v2.1",
                "robot_type": "elf3_sonic_full_body",
                "fps": PUBLISHED_FPS,
                "observation_state_dimension": 31,
                "action_wbc_dimension": 31,
                "motion_token_dimension": MOTION_TOKEN_DIM,
                "motion_token_source": (
                    "customer_sonic_step28800_post_fsq_decoder_input"
                ),
                "motion_token_is_offline_sidecar": False,
                "motion_token_is_online_same_inference_output": True,
                "gripper_state_joint_names": list(GRIPPER_STATE_JOINT_NAMES),
                "gripper_command_names": list(GRIPPER_COMMAND_NAMES),
            },
            "camera_contract": {
                "rgb": list(self.rgb_camera_names),
                "depth": DEPTH_CAMERA_NAME if self.depth_enabled else None,
                "rgb_storage": "JPEG, decoded RGB input",
                "depth_storage": "NumPy .npy preserving ROS scalar dtype",
                "mounts": self.camera_metadata,
            },
            "timebase": {
                "row_clock": "real_robot_control_step",
                "row_fps": self.fps,
                "timestamp_rule": "frame_index / row_fps",
                "wall_clock_is_diagnostic_only": True,
                "camera_alignment": "latest_hardware_frame_at_or_before_control_step",
                "source_header_stamps_preserved": True,
                "source_receive_offsets_preserved": True,
            },
        }
        if self.wrist_diagnostic:
            self.meta.pop('published_dataset_contract')
            self.meta['diagnostic_only'] = True
            self.meta['training_export_allowed'] = False
            self.meta['missing_modalities'] = ['head_depth']
            self.meta['gripper_coordinate'] = 'raw_motor_rad_not_training_coordinate'
            self.meta['gripper_identity_contract'] = self.gripper_identities
            self.meta['timebase']['uniform_training_timebase_verified'] = False
            self.meta['timebase'].update(row_clock='head_rgb_source_header',
                timestamp_rule='head_source_stamp - first_saved_head_source_stamp',
                row_fps_is_nominal_only=True, wall_clock_is_diagnostic_only=False,
                camera_alignment='buffered_nearest_source_header_within_25ms_no_reuse')
            if self.common_metadata.get('capture_profile') == 'sonic50_latest_rgb':
                self.meta['gripper_coordinate'] = 'absolute_motor_radians'
                self.meta['timebase'].update(row_clock='sonic_inference_source_stamp',
                    timestamp_rule='inference_source_stamp - first_saved_inference_source_stamp',
                    camera_alignment=('latest_received_at_collection_tick'
                        if self.common_metadata.get('sample_selection') == 'latest_received_v1' else
                        'latest_source_at_or_before_inference_max_age_100ms_reuse_allowed'))
        self._start_writer()
        return self.episode_id

    def _start_writer(self):
        self.worker = threading.Thread(target=self._worker, name="episode-writer", daemon=True)
        self.worker.start()

    def writer_queue_size(self):
        return self.queue.qsize()

    def _drain_writer(self):
        self.queue.join()
        self.stop_worker = True
        if self.worker is not None:
            self.worker.join(timeout=60.0)
            if self.worker.is_alive():
                raise RuntimeError("episode writer did not stop")

    def _prepare_rgb(self, image):
        return np.ascontiguousarray(image).copy()

    def _submit_frame(self, row, rgb, depth):
        try:
            self.queue.put_nowait(dict(row=row, rgb=rgb, depth=depth))
        except Full as exc:
            raise RuntimeError("episode writer queue is full; refusing to lose a frame") from exc

    def capture(
        self,
        *,
        anchor_sequence: int,
        observation: dict[str, np.ndarray],
        action: dict[str, np.ndarray],
        effective_continuous_action_29,
        diagnostic_action_history,
        motion_token,
        sonic_inference: dict[str, Any],
        projected_gravity,
        imu: dict[str, Any],
        rgb_images: dict[str, np.ndarray],
        depth_image: np.ndarray | None,
        depth_encoding: str | None,
        pico: dict,
        smpl_ref: dict,
        state_machine: dict,
        motion_command: dict | None,
        auxiliary: dict[str, Any],
        gripper: dict[str, Any],
        source_timestamps: dict[str, float],
        source_receive_offsets: dict[str, float],
    ) -> None:
        if not self.recording:
            return
        anchor_sequence = int(anchor_sequence)
        if anchor_sequence <= self.last_anchor_sequence:
            raise ValueError("source anchor sequence must be strictly increasing")
        joint_count = len(CANONICAL_JOINT_NAMES)
        observation_normalized = self._motor_group(
            observation, joint_count, "observation"
        )
        action_normalized = self._motor_group(action, joint_count, "action")
        token = validate_effective_continuous_action_29(effective_continuous_action_29)
        training_token = validate_motion_token(motion_token)
        inference_sequence = int(sonic_inference.get("sequence", -1))
        if inference_sequence <= self.last_sonic_inference_sequence:
            raise ValueError("Sonic inference sequence must be strictly increasing")
        model_id = str(sonic_inference.get("model_id", ""))
        model_sha256 = str(sonic_inference.get("model_sha256", "")).lower()
        if not model_id:
            raise ValueError("Sonic inference model_id is required")
        if len(model_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in model_sha256
        ):
            raise ValueError("Sonic inference model_sha256 is invalid")
        history = np.asarray(diagnostic_action_history, dtype=np.float32)
        if history.shape != (DIAGNOSTIC_ACTION_HISTORY, DIAGNOSTIC_ACTION_DIM):
            raise ValueError(
                "diagnostic_action_history must have shape "
                f"({DIAGNOSTIC_ACTION_HISTORY},{DIAGNOSTIC_ACTION_DIM}), got {history.shape}"
            )
        if not np.isfinite(history).all():
            raise ValueError("diagnostic_action_history contains NaN or Inf")
        gravity = finite_vector(projected_gravity, 3, "projected_gravity")
        if not np.isclose(np.linalg.norm(gravity), 1.0, rtol=0.0, atol=1.0e-4):
            raise ValueError("projected_gravity is not a unit vector")
        gripper_state = gripper.get("state")
        gripper_command = gripper.get("command")
        if self.wrist_diagnostic:
            validate_raw_row(gripper, self.gripper_identities)
        elif self.mode == "real":
            validate_gripper_row(gripper)
        if gripper_state is None or gripper_command is None:
            if self.mode == "real":
                raise ValueError(
                    "real rows require mapped 2-D gripper state and final 2-D "
                    "gripper command; trigger intent is not an admissible substitute"
                )
            dataset_state = None
            dataset_action = None
        elif self.wrist_diagnostic:
            dataset_state = None
            dataset_action = None
        else:
            dataset_state = pack_state_31(
                observation_normalized["position"][: len(BODY_JOINT_NAMES)],
                gripper_state,
            )
            dataset_action = pack_action_31(
                action_normalized["position"][: len(BODY_JOINT_NAMES)],
                gripper_command,
            )
        missing = [name for name in self.rgb_camera_names if name not in rgb_images]
        if missing:
            raise ValueError(f"missing RGB cameras: {missing}")
        normalized_rgb = {}
        for name in self.rgb_camera_names:
            image = np.asarray(rgb_images[name])
            if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
                raise ValueError(
                    f"{name} must be uint8 HxWx3 RGB, got {image.dtype} {image.shape}"
                )
            expected_shape = ((int(self.camera_metadata[name]['height']), int(self.camera_metadata[name]['width']), 3)
                              if self.wrist_diagnostic else (480, 640, 3))
            if self.mode == "real" and image.shape != expected_shape:
                raise ValueError(
                    f"{name} must be 640x480 RGB for formal data, got {image.shape}"
                )
            normalized_rgb[name] = self._prepare_rgb(image)
        depth = None
        if self.depth_enabled:
            if depth_image is None:
                raise ValueError("head depth is enabled but missing")
            depth = np.asarray(depth_image)
            if depth.ndim != 2 or depth.dtype not in (
                np.dtype("uint16"),
                np.dtype("float32"),
            ):
                raise ValueError(
                    "head depth must be uint16/float32 HxW, "
                    f"got {depth.dtype} {depth.shape}"
                )
            if np.issubdtype(depth.dtype, np.floating) and not np.isfinite(depth).all():
                raise ValueError("head depth contains NaN or Inf")
        index = self.frame_count
        source_stamp = float(auxiliary.get('anchor_stamp_unix', 0))
        if self.wrist_diagnostic and (not np.isfinite(source_stamp) or source_stamp <= 0
                or (self.last_source_stamp is not None and source_stamp <= self.last_source_stamp)):
            raise ValueError('diagnostic rows require strictly increasing source anchors')
        source_origin = self.first_source_stamp if self.first_source_stamp is not None else source_stamp
        paths = {
            name: f"{name}/rgb/{index:06d}.jpg" for name in self.rgb_camera_names
        }
        depth_description = None
        if depth is not None:
            depth_description = {
                "path": f"{DEPTH_CAMERA_NAME}/depth/{index:06d}.npy",
                "encoding": str(depth_encoding),
                "dtype": str(depth.dtype),
                "shape": list(depth.shape),
            }
        row = {
            "diagnostic_only": self.wrist_diagnostic,
            "idx": index,
            "t_episode": (source_stamp-source_origin if self.wrist_diagnostic else round(index / float(self.fps), 9)),
            "t_wall_episode": round(time.time() - self.started_unix, 9),
            "t_unix": round(time.time(), 9),
            "anchor_sequence": anchor_sequence,
            "observation.state": observation_normalized["position"],
            "observation.velocity": observation_normalized["velocity"],
            "observation.effort": observation_normalized["effort"],
            "observation.driver_temperature": observation_normalized[
                "driver_temperature"
            ],
            "observation.motor_temperature": observation_normalized[
                "motor_temperature"
            ],
            "observation.joint_present": observation_normalized["joint_present"],
            "observation.driver_temperature_present": observation_normalized[
                "driver_temperature_present"
            ],
            "observation.motor_temperature_present": observation_normalized[
                "motor_temperature_present"
            ],
            "observation.projected_gravity": gravity.tolist(),
            "dataset.observation.state": (
                dataset_state.tolist() if dataset_state is not None else None
            ),
            "observation.imu": imu,
            "action": action_normalized["position"],
            "action.velocity": action_normalized["velocity"],
            "action.torque": action_normalized["torque"],
            "action.kp": action_normalized["kp"],
            "action.kd": action_normalized["kd"],
            "action.joint_present": action_normalized["joint_present"],
            "dataset.action.wbc": (
                dataset_action.tolist() if dataset_action is not None else None
            ),
            "action.motion_token": training_token.tolist(),
            "sonic.inference": {
                **sonic_inference,
                "sequence": inference_sequence,
                "model_id": model_id,
                "model_sha256": model_sha256,
                "source": "same_run_online_post_fsq_decoder_input",
            },
            "sonic.effective_continuous_action_29": token.tolist(),
            "sonic.effective_continuous_action_history_10x29": history.tolist(),
            "pico.raw": pico,
            "pico.smpl_ref": smpl_ref,
            "state_machine": state_machine,
            "motion_command": motion_command,
            "auxiliary": auxiliary,
            "gripper": gripper,
            "observation.images": paths,
            "observation.depth": depth_description,
            "source_timestamps_unix": source_timestamps,
            "source_receive_offsets_s": source_receive_offsets,
        }
        self._submit_frame(row, normalized_rgb,
                           np.ascontiguousarray(depth).copy() if depth is not None else None)
        self.frame_count += 1
        if self.wrist_diagnostic:
            self.first_source_stamp = source_origin
            self.last_source_stamp = source_stamp
        self.last_anchor_sequence = anchor_sequence
        self.last_sonic_inference_sequence = inference_sequence

    @staticmethod
    def _motor_group(group: dict[str, np.ndarray], size: int, label: str) -> dict:
        result = {}
        for key in (
            "position",
            "velocity",
            "effort",
            "driver_temperature",
            "motor_temperature",
            "torque",
            "kp",
            "kd",
        ):
            if key in group:
                result[key] = finite_vector(group[key], size, f"{label}.{key}").tolist()
        for key in (
            "joint_present",
            "driver_temperature_present",
            "motor_temperature_present",
        ):
            if key in group:
                mask = np.asarray(group[key], dtype=bool).reshape(-1)
                if mask.shape != (size,):
                    raise ValueError(f"{label}.{key} must have shape ({size},)")
                result[key] = mask.tolist()
        required = (
            ("position", "velocity", "effort", "joint_present")
            if label == "observation"
            else ("position", "velocity", "torque", "kp", "kd", "joint_present")
        )
        missing = [key for key in required if key not in result]
        if missing:
            raise ValueError(f"{label} missing motor fields {missing}")
        if label == "observation":
            for key in ("driver_temperature", "motor_temperature"):
                result.setdefault(key, [0.0] * size)
                result.setdefault(f"{key}_present", [False] * size)
        return result

    def end_episode(self, termination: dict[str, Any], min_frames: int) -> dict[str, Any]:
        if not self.recording:
            return {}
        # Keep the episode active when the operator asks to finish too early.
        # The node normally performs the same preflight check, but preserving
        # this invariant here prevents direct callers from stranding an
        # in-progress episode with ``recording`` already set to false.
        if self.frame_count < int(min_frames):
            raise RuntimeError(
                f"episode has only {self.frame_count} rows; minimum is {int(min_frames)}"
            )
        self.recording = False
        self._drain_writer()
        if self.errors:
            error_text = "writer errors: " + "; ".join(self.errors[:5])
            self.recording = True
            preserved = self.preserve_interrupted_episode('writer_error', min_frames=min_frames)
            raise RuntimeError(error_text + '; preserved=' + str(preserved['path']))
        if len(self.rows) != self.frame_count:
            self.recording = True
            preserved = self.preserve_interrupted_episode('writer_count_mismatch', min_frames=min_frames)
            raise RuntimeError('frame count mismatch; preserved=' + str(preserved['path']))
        assert self.inprogress_dir is not None and self.final_dir is not None
        self.rows.sort(key=lambda row: int(row["idx"]))
        frames_tmp = self.inprogress_dir / "frames.jsonl.tmp"
        with frames_tmp.open("w", encoding="utf-8") as handle:
            for row in self.rows:
                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(frames_tmp, self.inprogress_dir / "frames.jsonl")
        finished = time.time()
        self.meta.update(
            {
                "end_time_unix": finished,
                "end_time_utc": utc_now(),
                "duration_s": (self.last_source_stamp-self.first_source_stamp if self.wrist_diagnostic
                               else round(self.frame_count / float(self.fps), 9)),
                "wall_duration_s": round(finished - self.started_unix, 9),
                "num_frames": self.frame_count,
                "termination": dict(termination),
            }
        )
        if self.wrist_diagnostic:
            duration = self.last_source_stamp-self.first_source_stamp
            self.meta['actual_row_rate_hz'] = (self.frame_count-1)/duration if duration > 0 else 0.
        meta_tmp = self.inprogress_dir / "meta.json.tmp"
        with meta_tmp.open("w", encoding="utf-8") as handle:
            json.dump(self.meta, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(meta_tmp, self.inprogress_dir / "meta.json")
        os.replace(self.inprogress_dir, self.final_dir)
        summary = {
            "episode_id": self.episode_id,
            "frames": self.frame_count,
            "path": str(self.final_dir),
        }
        self.episode_id += 1
        self.inprogress_dir = None
        self.final_dir = None
        return summary

    def preserve_interrupted_episode(
        self, reason: str, *, min_frames: int, discarded: bool = False
    ) -> dict[str, Any]:
        """Persist a short/incomplete episode outside the exportable episode set.

        A normal Ctrl-C or managed process shutdown must not silently erase
        frames that have already reached the writer.  Interrupted episodes are
        intentionally placed below ``_interrupted`` so the strict exporter,
        which scans ``episode_*`` directly under ``source_root``, cannot mistake
        them for complete training episodes.
        """

        if not self.recording:
            return {"preserved": False, "reason": reason}
        self.recording = False
        self._drain_writer()
        assert self.inprogress_dir is not None

        self.rows.sort(key=lambda row: int(row["idx"]))
        frames_tmp = self.inprogress_dir / "frames.jsonl.tmp"
        with frames_tmp.open("w", encoding="utf-8") as handle:
            for row in self.rows:
                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(frames_tmp, self.inprogress_dir / "frames.jsonl")

        finished = time.time()
        saved_frames = len(self.rows)
        self.meta.update(
            {
                "end_time_unix": finished,
                "end_time_utc": utc_now(),
                "duration_s": ((self.rows[-1]['t_episode']-self.rows[0]['t_episode'] if self.rows else 0.)
                               if self.wrist_diagnostic else round(saved_frames / float(self.fps), 9)),
                "wall_duration_s": round(finished - self.started_unix, 9),
                "num_frames": saved_frames,
                "incomplete": True,
                "discarded": discarded,
                "excluded_from_training": True,
                "termination": {
                    "success": False,
                    "reason": str(reason),
                    "minimum_frames": int(min_frames),
                    "accepted_frames": self.frame_count,
                    "saved_frames": saved_frames,
                    "writer_errors": list(self.errors),
                },
            }
        )
        meta_tmp = self.inprogress_dir / "meta.json.tmp"
        with meta_tmp.open("w", encoding="utf-8") as handle:
            json.dump(self.meta, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(meta_tmp, self.inprogress_dir / "meta.json")

        # Operator discard follows the upstream semantics: retain but exclude.
        # Faults remain distinctly _interrupted, never counted as user discard.
        interrupted_root = self.source_root / ("_discarded" if discarded else "_interrupted")
        interrupted_root.mkdir(parents=True, exist_ok=True)
        timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        preserved_dir = interrupted_root / f"episode_{self.episode_id:04d}_{timestamp}"
        suffix = 1
        while preserved_dir.exists():
            preserved_dir = interrupted_root / (
                f"episode_{self.episode_id:04d}_{timestamp}_{suffix}"
            )
            suffix += 1
        os.replace(self.inprogress_dir, preserved_dir)
        result = {
            "preserved": True,
            "episode_id": self.episode_id,
            "accepted_frames": self.frame_count,
            "saved_frames": saved_frames,
            "reason": reason,
            "path": str(preserved_dir),
        }
        self.episode_id += 1
        self.inprogress_dir = None
        self.final_dir = None
        return result

    def discard_episode(self, reason: str) -> dict[str, Any]:
        if not self.recording:
            return {"discarded": False, "reason": reason}
        self.recording = False
        self.discarding = True
        self.queue.join()
        self.stop_worker = True
        if self.worker is not None:
            self.worker.join(timeout=60.0)
        path = self.inprogress_dir
        if path is not None and path.exists():
            shutil.rmtree(path)
        result = {
            "discarded": True,
            "episode_id": self.episode_id,
            "frames": self.frame_count,
            "reason": reason,
        }
        self.inprogress_dir = None
        self.final_dir = None
        self.discarding = False
        return result

    def _worker(self) -> None:
        # Keep at most one encoded image per camera, reset for every episode.
        # Numeric rows keep distinct paths; only identical source images skip encoding.
        jpeg_cache = {}
        reuse_jpeg = self.common_metadata.get('capture_profile') == 'sonic50_latest_rgb'
        while not self.stop_worker or not self.queue.empty():
            try:
                item = self.queue.get(timeout=0.2)
            except Empty:
                continue
            try:
                if self.discarding:
                    continue
                assert self.inprogress_dir is not None
                row = item["row"]
                index = int(row["idx"])
                for camera, rgb in item["rgb"].items():
                    path = self.inprogress_dir / camera / "rgb" / f"{index:06d}.jpg"
                    if reuse_jpeg:
                        aux = row['auxiliary']
                        key = (aux.get('capture_segment', 0),
                               aux['selected_camera_sequences'][camera],
                               row['source_timestamps_unix'][camera])
                        cached = jpeg_cache.get(camera)
                        if cached is None or cached[0] != key:
                            ok, encoded = cv2.imencode('.jpg', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                                [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
                            if not ok:
                                raise OSError(f"failed to encode {path}")
                            cached = (key, encoded.tobytes())
                            jpeg_cache[camera] = cached
                        path.write_bytes(cached[1])
                    else:
                        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                        if not cv2.imwrite(str(path), bgr,
                                           [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]):
                            raise OSError(f"failed to save {path}")
                if item["depth"] is not None:
                    depth_path = (
                        self.inprogress_dir
                        / DEPTH_CAMERA_NAME
                        / "depth"
                        / f"{index:06d}.npy"
                    )
                    with depth_path.open("wb") as handle:
                        np.save(handle, item["depth"], allow_pickle=False)
                self.rows.append(row)
            except Exception as exc:
                self.errors.append(
                    f"frame {item.get('row', {}).get('idx', '?')}: {exc}"
                )
            finally:
                self.queue.task_done()
