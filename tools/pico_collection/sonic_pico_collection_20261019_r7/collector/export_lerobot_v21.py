#!/usr/bin/env python3
"""Export ELF3 raw episodes to the approved 30 FPS LeRobot v2.1 contract."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

import cv2
import numpy as np

from attach_motion_tokens import (
    TOKEN_SOURCE,
    validate_token_array,
)
from real_robot_collection.contracts import (
    MOTION_TOKEN_DIM,
    PUBLISHED_FPS,
    STATE_DIM,
    WBC_ACTION_DIM,
)
from validate_raw import ValidationError, read_json, read_rows, source_path, validate_episode


TARGET_CODEBASE_VERSION = "v2.1"
ROBOT_TYPE = "elf3_sonic_full_body"
CHUNKS_SIZE = 1000
CAMERA_MAP = {
    "head_rgb": "observation.images.ego_view",
    "left_wrist_rgb": "observation.images.left_wrist",
    "right_wrist_rgb": "observation.images.right_wrist",
}
NUMERIC_KEYS = (
    "observation.state",
    "observation.projected_gravity",
    "action.wbc",
    "action.motion_token",
    "timestamp",
    "frame_index",
    "episode_index",
    "index",
    "task_index",
)


class ExportError(RuntimeError):
    pass


@dataclass(frozen=True)
class EpisodePlan:
    path: Path
    task: str
    rows: tuple[dict[str, Any], ...]
    motion_tokens: np.ndarray


@dataclass(frozen=True)
class ExportPlan:
    source_root: Path
    episodes: tuple[EpisodePlan, ...]
    fps: int
    total_frames: int
    sonic_model_sha256: str

    def lerobot_features(self) -> dict[str, dict[str, Any]]:
        result = {
            "observation.state": {"dtype": "float64", "shape": (31,), "names": None},
            "observation.projected_gravity": {"dtype": "float64", "shape": (3,), "names": ["x", "y", "z"]},
            "action.wbc": {"dtype": "float64", "shape": (31,), "names": None},
            "action.motion_token": {"dtype": "float64", "shape": (64,), "names": None},
        }
        for key in ("timestamp", "frame_index", "episode_index", "index", "task_index"):
            result[key] = {
                "dtype": "float32" if key == "timestamp" else "int64",
                "shape": (1,),
                "names": None,
            }
        for key in CAMERA_MAP.values():
            result[key] = {
                "dtype": "video",
                "shape": (480, 640, 3),
                "names": ["height", "width", "channel"],
                "info": {
                    "video.height": 480,
                    "video.width": 640,
                    "video.channels": 3,
                    "video.fps": PUBLISHED_FPS,
                    "video.codec": "h264",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "has_audio": False,
                },
            }
        return result


def _load_motion_tokens(
    episode: Path, rows: tuple[dict[str, Any], ...]
) -> tuple[np.ndarray, str]:
    tokens = validate_token_array(
        np.asarray([row.get("action.motion_token") for row in rows]),
        len(rows),
    )
    model_checksums = {
        str(row.get("sonic.inference", {}).get("model_sha256", "")).lower()
        for row in rows
    }
    if len(model_checksums) != 1:
        raise ExportError(f"{episode}: online tokens use multiple Sonic models")
    model_sha256 = next(iter(model_checksums))
    if len(model_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in model_sha256
    ):
        raise ExportError(f"{episode}: invalid online Sonic model checksum")
    return tokens, model_sha256


def _validate_dataset_row(row: dict[str, Any], episode: Path, index: int) -> None:
    expected = {
        "dataset.observation.state": (STATE_DIM,),
        "observation.projected_gravity": (3,),
        "dataset.action.wbc": (WBC_ACTION_DIM,),
    }
    for key, shape in expected.items():
        value = np.asarray(row.get(key), dtype=np.float64)
        if value.shape != shape or not np.isfinite(value).all():
            raise ExportError(f"{episode}: row {index} invalid {key}: {value.shape}")
    images = row.get("observation.images")
    if not isinstance(images, dict) or tuple(images) != tuple(CAMERA_MAP):
        raise ExportError(f"{episode}: row {index} does not contain the three fixed cameras")


def build_plan(
    source: Path, *, formal: bool = True, max_episodes: int | None = None
) -> ExportPlan:
    root = source_path(source)
    if not root.is_dir():
        raise ExportError(f"source directory does not exist: {root}")
    if sorted(root.glob(".episode_*.inprogress")):
        raise ExportError(f"incomplete source episode exists in {root}")
    paths = sorted(path for path in root.glob("episode_*") if path.is_dir())
    if max_episodes is not None:
        if max_episodes <= 0:
            raise ExportError("--max-episodes must be positive")
        paths = paths[:max_episodes]
    if not paths:
        raise ExportError(f"no episodes in {root}")
    episodes: list[EpisodePlan] = []
    model_checksums: set[str] = set()
    for path in paths:
        validate_episode(path, decode_all_images=False, formal=formal)
        meta = read_json(path / "meta.json")
        if int(meta.get("fps", -1)) != PUBLISHED_FPS:
            raise ExportError(f"{path}: fps must be {PUBLISHED_FPS}")
        rows = tuple(read_rows(path / "frames.jsonl"))
        wall_stamps = np.asarray([row.get("source_timestamps_unix", {}).get("head_rgb", float("nan")) for row in rows])
        # Reject silent time compression: 30-Hz labels cannot hide missing
        # camera frames. Keep the interrupted raw episode for diagnosis.
        if len(wall_stamps) > 1:
            intervals = np.diff(wall_stamps)
            if (not np.isfinite(intervals).all() or np.any(intervals <= 0)
                    or np.max(intervals) > 1.5/PUBLISHED_FPS
                    or not .9/PUBLISHED_FPS <= np.mean(intervals) <= 1.1/PUBLISHED_FPS):
                raise ExportError(f"{path}: actual camera timebase is not continuous 30Hz; no relabeling allowed")
        if len(rows) < 100:
            raise ExportError(f"{path}: formal episodes need at least 100 rows for stats")
        for index, row in enumerate(rows):
            _validate_dataset_row(row, path, index)
        tokens, model_sha256 = _load_motion_tokens(path, rows)
        model_checksums.add(model_sha256)
        episodes.append(
            EpisodePlan(path, str(meta["task_name"]), rows, tokens)
        )
    if len(model_checksums) != 1:
        raise ExportError("all episodes in one dataset must use the same Sonic model")
    return ExportPlan(
        root,
        tuple(episodes),
        PUBLISHED_FPS,
        sum(len(x.rows) for x in episodes),
        next(iter(model_checksums)),
    )


def frame_from_row(
    episode: Path,
    row: dict[str, Any],
    plan: ExportPlan,
    *,
    motion_token: np.ndarray | None = None,
) -> dict[str, Any]:
    """Return only train-contract features (images decoded as RGB)."""

    if motion_token is None:
        raise ExportError("frame_from_row requires the aligned 64-D motion token")
    frame: dict[str, Any] = {
        "observation.state": np.asarray(row["dataset.observation.state"], dtype=np.float64),
        "observation.projected_gravity": np.asarray(row["observation.projected_gravity"], dtype=np.float64),
        "action.wbc": np.asarray(row["dataset.action.wbc"], dtype=np.float64),
        "action.motion_token": np.asarray(motion_token, dtype=np.float64),
    }
    for source_name, output_key in CAMERA_MAP.items():
        image = cv2.imread(
            str(episode / row["observation.images"][source_name]), cv2.IMREAD_COLOR
        )
        if image is None or image.shape != (480, 640, 3):
            raise ExportError(f"cannot decode 640x480 RGB image for {source_name}")
        frame[output_key] = np.ascontiguousarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    return frame


def _require_pyarrow():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except Exception as exc:
        raise ExportError(
            "pyarrow is required; install requirements-export.txt in the export environment"
        ) from exc
    return pa, pq


def _hf_metadata() -> bytes:
    features = {
        key: (
            {
                "feature": {"dtype": "float64", "_type": "Value"},
                "length": size,
                "_type": "Sequence",
            }
            if size is not None
            else {"dtype": dtype, "_type": "Value"}
        )
        for key, dtype, size in (
            ("observation.state", "float64", 31),
            ("observation.projected_gravity", "float64", 3),
            ("action.wbc", "float64", 31),
            ("action.motion_token", "float64", 64),
            ("timestamp", "float32", None),
            ("frame_index", "int64", None),
            ("episode_index", "int64", None),
            ("index", "int64", None),
            ("task_index", "int64", None),
        )
    }
    return json.dumps({"info": {"features": features}}, separators=(",", ":")).encode()


def _write_parquet(
    path: Path,
    episode: EpisodePlan,
    episode_index: int,
    first_global_index: int,
    task_index: int,
) -> dict[str, np.ndarray]:
    pa, pq = _require_pyarrow()
    count = len(episode.rows)
    arrays = {
        "observation.state": np.asarray([row["dataset.observation.state"] for row in episode.rows], dtype=np.float64),
        "observation.projected_gravity": np.asarray([row["observation.projected_gravity"] for row in episode.rows], dtype=np.float64),
        "action.wbc": np.asarray([row["dataset.action.wbc"] for row in episode.rows], dtype=np.float64),
        "action.motion_token": np.asarray(episode.motion_tokens, dtype=np.float64),
        "timestamp": np.arange(count, dtype=np.float32) / np.float32(PUBLISHED_FPS),
        "frame_index": np.arange(count, dtype=np.int64),
        "episode_index": np.full(count, episode_index, dtype=np.int64),
        "index": np.arange(first_global_index, first_global_index + count, dtype=np.int64),
        "task_index": np.full(count, task_index, dtype=np.int64),
    }
    schema = pa.schema(
        [
            pa.field("observation.state", pa.list_(pa.float64(), list_size=31)),
            pa.field("observation.projected_gravity", pa.list_(pa.float64(), list_size=3)),
            pa.field("action.wbc", pa.list_(pa.float64(), list_size=31)),
            pa.field("action.motion_token", pa.list_(pa.float64(), list_size=64)),
            pa.field("timestamp", pa.float32()),
            pa.field("frame_index", pa.int64()),
            pa.field("episode_index", pa.int64()),
            pa.field("index", pa.int64()),
            pa.field("task_index", pa.int64()),
        ],
        metadata={b"huggingface": _hf_metadata()},
    )
    columns = {
        key: value.tolist() if value.ndim == 2 else value
        for key, value in arrays.items()
    }
    table = pa.Table.from_pydict(columns, schema=schema)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".parquet.tmp")
    pq.write_table(table, tmp, compression="zstd", row_group_size=128, version="2.6")
    reopened = pq.read_table(tmp)
    if reopened.num_rows != count or reopened.schema != schema:
        raise ExportError(f"Parquet verification failed before commit: {tmp}")
    with tmp.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    return arrays


class RunningStats:
    def __init__(self, feature_shape: tuple[int, ...]) -> None:
        self.shape = feature_shape
        self.minimum = np.full(feature_shape, np.inf, dtype=np.float64)
        self.maximum = np.full(feature_shape, -np.inf, dtype=np.float64)
        self.total = np.zeros(feature_shape, dtype=np.float64)
        self.total_square = np.zeros(feature_shape, dtype=np.float64)
        self.observations = 0
        self.logical_count = 0

    def update(self, values: np.ndarray, *, logical_count: int | None = None) -> None:
        value = np.asarray(values, dtype=np.float64).reshape((-1, *self.shape))
        self.minimum = np.minimum(self.minimum, value.min(axis=0))
        self.maximum = np.maximum(self.maximum, value.max(axis=0))
        self.total += value.sum(axis=0)
        self.total_square += np.square(value).sum(axis=0)
        self.observations += value.shape[0]
        self.logical_count += value.shape[0] if logical_count is None else logical_count

    def result(self) -> dict[str, list]:
        mean = self.total / self.observations
        variance = np.maximum(self.total_square / self.observations - np.square(mean), 0.0)
        return {
            "min": self.minimum.tolist(),
            "max": self.maximum.tolist(),
            "mean": mean.tolist(),
            "std": np.sqrt(variance).tolist(),
            "count": [int(self.logical_count)],
        }


def _episode_stats(
    arrays: dict[str, np.ndarray], episode: EpisodePlan
) -> tuple[dict[str, dict], dict[str, RunningStats]]:
    accumulators: dict[str, RunningStats] = {}
    for key, value in arrays.items():
        reshaped = value.reshape(len(episode.rows), -1)
        accumulator = RunningStats((reshaped.shape[1],))
        accumulator.update(reshaped)
        accumulators[key] = accumulator
    sample_indices = np.linspace(0, len(episode.rows) - 1, 100, dtype=np.int64)
    for source_name, output_key in CAMERA_MAP.items():
        accumulator = RunningStats((3, 1, 1))
        for index in sample_indices:
            path = episode.path / episode.rows[int(index)]["observation.images"][source_name]
            bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if bgr is None or bgr.shape != (480, 640, 3):
                raise ExportError(f"invalid image while computing stats: {path}")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0
            accumulator.update(rgb.reshape(-1, 3)[:, :, None, None], logical_count=0)
        accumulator.logical_count = 100
        accumulators[output_key] = accumulator
    return {key: value.result() for key, value in accumulators.items()}, accumulators


def _merge_stats(target: dict[str, RunningStats], source: dict[str, RunningStats]) -> None:
    for key, item in source.items():
        if key not in target:
            target[key] = RunningStats(item.shape)
        output = target[key]
        output.minimum = np.minimum(output.minimum, item.minimum)
        output.maximum = np.maximum(output.maximum, item.maximum)
        output.total += item.total
        output.total_square += item.total_square
        output.observations += item.observations
        output.logical_count += item.logical_count


def _verify_video(path: Path, expected_frames: int) -> None:
    command = [
        "ffprobe", "-v", "error",
        "-show_entries", "stream=codec_type,codec_name,pix_fmt,width,height,r_frame_rate,avg_frame_rate,nb_frames",
        "-of", "json", str(path),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise ExportError(f"ffprobe failed for {path}: {result.stderr.strip()}")
    streams = json.loads(result.stdout).get("streams", [])
    if len(streams) != 1:
        raise ExportError(f"{path}: expected exactly one video stream")
    stream = streams[0]
    expected = {
        "codec_type": "video", "codec_name": "h264", "pix_fmt": "yuv420p", "width": 640,
        "height": 480,
        "r_frame_rate": f"{PUBLISHED_FPS}/1",
        "avg_frame_rate": f"{PUBLISHED_FPS}/1",
        "nb_frames": str(expected_frames),
    }
    mismatches = {
        key: (stream.get(key), value)
        for key, value in expected.items()
        if stream.get(key) != value
    }
    if mismatches:
        raise ExportError(f"{path}: encoded video contract mismatch {mismatches}")


def _copy_jpegs_and_encode_video(
    staging: Path, episode: EpisodePlan, episode_index: int
) -> None:
    chunk = episode_index // CHUNKS_SIZE
    for source_name, output_key in CAMERA_MAP.items():
        image_dir = staging / "images" / output_key / f"episode_{episode_index:06d}"
        image_dir.mkdir(parents=True, exist_ok=True)
        for frame_index, row in enumerate(episode.rows):
            source_image = episode.path / row["observation.images"][source_name]
            shutil.copyfile(source_image, image_dir / f"frame_{frame_index:06d}.jpg")
        video = staging / "videos" / f"chunk-{chunk:03d}" / output_key / f"episode_{episode_index:06d}.mp4"
        video.parent.mkdir(parents=True, exist_ok=True)
        tmp = video.with_name(f".{video.name}.tmp.mp4")
        command = [
            "ffmpeg", "-v", "error", "-y", "-framerate", str(PUBLISHED_FPS),
            "-i", str(image_dir / "frame_%06d.jpg"), "-an", "-c:v", "libx264",
            "-pix_fmt", "yuv420p", "-r", str(PUBLISHED_FPS), str(tmp),
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            raise ExportError(f"ffmpeg failed for {output_key}: {result.stderr.strip()}")
        _verify_video(tmp, len(episode.rows))
        os.replace(tmp, video)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _run_staging_validator(staging: Path) -> None:
    validator = Path(__file__).resolve().with_name("validate_lerobot_v21.py")
    result = subprocess.run(
        [sys.executable, str(validator), str(staging)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise ExportError(f"staging dataset validation failed: {detail}")


def _modality() -> dict:
    return {
        "state": {
            "left_leg": {"start": 3, "end": 9},
            "right_leg": {"start": 9, "end": 15},
            "waist": {"start": 0, "end": 3},
            "left_arm": {"start": 15, "end": 22},
            "left_gripper": {"start": 29, "end": 30},
            "right_arm": {"start": 22, "end": 29},
            "right_gripper": {"start": 30, "end": 31},
            "projected_gravity": {"start": 0, "end": 3, "original_key": "observation.projected_gravity"},
        },
        "action": {
            "motion_token": {"start": 0, "end": 64, "original_key": "action.motion_token"},
            "left_gripper": {"start": 29, "end": 30, "original_key": "action.wbc"},
            "right_gripper": {"start": 30, "end": 31, "original_key": "action.wbc"},
        },
        "video": {
            "ego_view": {"original_key": "observation.images.ego_view"},
            "left_wrist": {"original_key": "observation.images.left_wrist"},
            "right_wrist": {"original_key": "observation.images.right_wrist"},
        },
        "annotation": {"human.task_description": {"original_key": "task_index"}},
    }


def export_dataset(plan: ExportPlan, *, output: Path, repo_id: str = "local/elf3_collection", **_: Any) -> None:
    output = output.expanduser().resolve()
    if output.exists():
        raise ExportError(f"output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.inprogress"
    if staging.exists():
        raise ExportError(f"stale incomplete export exists: {staging}")
    staging.mkdir()
    task_names = list(dict.fromkeys(episode.task for episode in plan.episodes))
    task_indices = {task: index for index, task in enumerate(task_names)}
    episode_rows: list[dict] = []
    episode_stats_rows: list[dict] = []
    global_stats: dict[str, RunningStats] = {}
    global_index = 0
    try:
        for episode_index, episode in enumerate(plan.episodes):
            chunk = episode_index // CHUNKS_SIZE
            parquet = staging / "data" / f"chunk-{chunk:03d}" / f"episode_{episode_index:06d}.parquet"
            arrays = _write_parquet(
                parquet, episode, episode_index, global_index, task_indices[episode.task]
            )
            _copy_jpegs_and_encode_video(staging, episode, episode_index)
            stats, accumulators = _episode_stats(arrays, episode)
            _merge_stats(global_stats, accumulators)
            episode_rows.append({"episode_index": episode_index, "tasks": [episode.task], "length": len(episode.rows)})
            episode_stats_rows.append({"episode_index": episode_index, "stats": stats})
            global_index += len(episode.rows)
            print(f"EXPORTED {episode.path.name}: episode={episode_index} frames={len(episode.rows)}")
        meta = staging / "meta"
        features = plan.lerobot_features()
        info = {
            "codebase_version": TARGET_CODEBASE_VERSION,
            "robot_type": ROBOT_TYPE,
            "total_episodes": len(plan.episodes),
            "total_frames": plan.total_frames,
            "total_tasks": len(task_names),
            "total_videos": len(plan.episodes) * len(CAMERA_MAP),
            "total_chunks": (len(plan.episodes) + CHUNKS_SIZE - 1) // CHUNKS_SIZE,
            "chunks_size": CHUNKS_SIZE,
            "fps": PUBLISHED_FPS,
            "splits": {"train": f"0:{len(plan.episodes)}"},
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            "features": features,
            "timebase": {
                "row_clock": "real_robot_control_step",
                "row_fps": PUBLISHED_FPS,
                "timestamp_rule": "frame_index / row_fps",
                "wall_clock_is_diagnostic_only": True,
                "camera_alignment": "latest_hardware_frame_at_or_before_control_step",
                "camera_sequence_scope": "episode",
            },
            "embodiment": {
                "tag": "ELF3_NATIVE_1D", "name": "ELF3_SONIC_NATIVE_1D", "model_action_dimension": 66,
                "purpose": "vla_training_and_inference", "sonic_rl_training_data": False,
                "state_dimension": 31, "body_motion_token_dimension": 64,
                "value": "elf3_native_1d", "projector_slot": 14,
                "timebase": "30Hz_real_capture_not_50Hz_legacy",
                "left_gripper_action_key": "left_gripper", "right_gripper_action_key": "right_gripper",
                "decoder_compatibility": "customer_elf3_step28800_only_not_unitree_g1_sonic",
            },
            "motion_token": {
                "key": "action.motion_token", "dimension": 64, "dtype": "float64",
                "source": TOKEN_SOURCE, "fsq_levels": 32,
                "extraction": "online_same_sonic_inference_as_action",
                "sonic_model_sha256": plan.sonic_model_sha256,
            },
        }
        _write_json(meta / "info.json", info)
        _write_json(meta / "modality.json", _modality())
        _write_json(meta / "native66_contract.json", {
            "embodiment": "elf3_native_1d", "stored_state": 31,
            "stored_wbc": 31, "model_action": 66, "fps": PUBLISHED_FPS,
            "legacy_50hz_checkpoint_direct_temporal_compatibility": False,
            "gripper_mappings": [episode.rows[0]["gripper"]["mapping"] for episode in plan.episodes],
        })
        _write_jsonl(meta / "tasks.jsonl", [{"task_index": i, "task": task} for i, task in enumerate(task_names)])
        _write_jsonl(meta / "episodes.jsonl", episode_rows)
        _write_jsonl(meta / "episodes_stats.jsonl", episode_stats_rows)
        _write_json(meta / "stats.json", {key: value.result() for key, value in global_stats.items()})
        _run_staging_validator(staging)
        os.replace(staging, output)
    except Exception:
        # Keep the staging directory for diagnosis; it never looks committed.
        raise
    print(f"PASS: strict LeRobot {TARGET_CODEBASE_VERSION} written to {output}")


def print_plan(plan: ExportPlan) -> None:
    print(f"SOURCE OK: episodes={len(plan.episodes)} frames={plan.total_frames} fps={plan.fps} root={plan.source_root}")
    for key, feature in plan.lerobot_features().items():
        print(f"  {key}: {feature['dtype']}{feature['shape']}")


def main() -> int:
    parser = argparse.ArgumentParser(description="export simulation-compatible ELF3 LeRobot v2.1")
    parser.add_argument("source", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--repo-id", default="local/elf3_collection")
    parser.add_argument("--allow-debug-source", action="store_true")
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        plan = build_plan(
            args.source,
            formal=not args.allow_debug_source,
            max_episodes=args.max_episodes,
        )
        print_plan(plan)
        if args.dry_run:
            print("DRY RUN PASS: no output was written")
            return 0
        root = source_path(args.source)
        output = args.output_dir or (root.parent / "lerobot_v21")
        export_dataset(plan, output=output, repo_id=args.repo_id)
    except (ExportError, ValidationError, OSError, ValueError, KeyError) as exc:
        print(f"EXPORT ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
