#!/usr/bin/env python3
"""Independent validator for the strict ELF3 30 FPS dataset contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np

from export_lerobot_v21 import CAMERA_MAP, CHUNKS_SIZE, NUMERIC_KEYS, ROBOT_TYPE
from real_robot_collection.contracts import PUBLISHED_FPS


class DatasetValidationError(RuntimeError):
    pass


def _json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise DatasetValidationError(f"cannot parse {path}: {exc}") from exc


def _jsonl(path: Path) -> list[dict]:
    try:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except Exception as exc:
        raise DatasetValidationError(f"cannot parse {path}: {exc}") from exc


def _probe_video(path: Path, expected_frames: int) -> None:
    command = [
        "ffprobe", "-v", "error",
        "-show_entries", "stream=codec_type,codec_name,pix_fmt,width,height,r_frame_rate,avg_frame_rate,nb_frames",
        "-of", "json", str(path),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise DatasetValidationError(f"ffprobe failed for {path}: {result.stderr.strip()}")
    streams = json.loads(result.stdout).get("streams", [])
    if len(streams) != 1:
        raise DatasetValidationError(f"{path}: expected one video stream")
    stream = streams[0]
    expected = {
        "codec_type": "video", "codec_name": "h264", "pix_fmt": "yuv420p", "width": 640,
        "height": 480,
        "r_frame_rate": f"{PUBLISHED_FPS}/1",
        "avg_frame_rate": f"{PUBLISHED_FPS}/1",
        "nb_frames": str(expected_frames),
    }
    for key, value in expected.items():
        if stream.get(key) != value:
            raise DatasetValidationError(
                f"{path}: {key}={stream.get(key)!r}, expected {value!r}"
            )


def validate_dataset(root: Path, *, decode_all_images: bool = False) -> dict:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except Exception as exc:
        raise DatasetValidationError("pyarrow is required to validate Parquet") from exc
    root = root.expanduser().resolve()
    for name in ("data", "videos", "images", "meta"):
        if not (root / name).is_dir():
            raise DatasetValidationError(f"missing directory: {root / name}")
    for name in ("info.json", "modality.json", "tasks.jsonl", "episodes.jsonl", "episodes_stats.jsonl", "stats.json"):
        if not (root / "meta" / name).is_file():
            raise DatasetValidationError(f"missing metadata: {name}")
    info = _json(root / "meta" / "info.json")
    if info.get("codebase_version") != "v2.1" or info.get("robot_type") != ROBOT_TYPE:
        raise DatasetValidationError("info.json version/robot_type mismatch")
    if int(info.get("fps", -1)) != PUBLISHED_FPS:
        raise DatasetValidationError("info.json fps mismatch")
    if info.get("embodiment", {}).get("tag") != "ELF3_NATIVE_1D":
        raise DatasetValidationError("native66 embodiment required")
    modality = _json(root / "meta" / "modality.json")
    expected_action = {"motion_token": (0,64,"action.motion_token"),
                       "left_gripper": (29,30,"action.wbc"),
                       "right_gripper": (30,31,"action.wbc")}
    for key,(start,end,original) in expected_action.items():
        if modality.get("action",{}).get(key) != {"start":start,"end":end,"original_key":original}:
            raise DatasetValidationError(f"native66 action modality mismatch: {key}")
    for key,start in (("left_gripper",29),("right_gripper",30)):
        if modality.get("state",{}).get(key) != {"start":start,"end":start+1}:
            raise DatasetValidationError(f"native66 state modality mismatch: {key}")
    for key,width in (("observation.state",31),("action.wbc",31),("action.motion_token",64)):
        if info.get("features",{}).get(key,{}).get("shape") != [width]:
            raise DatasetValidationError(f"native66 feature shape mismatch: {key}")
    episodes = _jsonl(root / "meta" / "episodes.jsonl")
    tasks = _jsonl(root / "meta" / "tasks.jsonl")
    if len(episodes) != int(info.get("total_episodes", -1)):
        raise DatasetValidationError("episode metadata count mismatch")
    if [row.get("episode_index") for row in episodes] != list(range(len(episodes))):
        raise DatasetValidationError("episode indices are not contiguous")
    task_ids = {int(row["task_index"]) for row in tasks}
    expected_schema = pa.schema(
        [
            pa.field("observation.state", pa.list_(pa.float64(), list_size=31)),
            pa.field("observation.projected_gravity", pa.list_(pa.float64(), list_size=3)),
            pa.field("action.wbc", pa.list_(pa.float64(), list_size=31)),
            pa.field("action.motion_token", pa.list_(pa.float64(), list_size=64)),
            pa.field("timestamp", pa.float32()), pa.field("frame_index", pa.int64()),
            pa.field("episode_index", pa.int64()), pa.field("index", pa.int64()),
            pa.field("task_index", pa.int64()),
        ]
    )
    next_global_index = 0
    total_frames = 0
    for episode_index, episode in enumerate(episodes):
        length = int(episode["length"])
        if length <= 0:
            raise DatasetValidationError(f"episode {episode_index} is empty")
        chunk = episode_index // CHUNKS_SIZE
        parquet = root / "data" / f"chunk-{chunk:03d}" / f"episode_{episode_index:06d}.parquet"
        table = pq.read_table(parquet)
        if table.schema.remove_metadata() != expected_schema:
            raise DatasetValidationError(f"{parquet}: schema mismatch: {table.schema}")
        if b"huggingface" not in (table.schema.metadata or {}):
            raise DatasetValidationError(f"{parquet}: missing Hugging Face metadata")
        if table.num_rows != length:
            raise DatasetValidationError(f"{parquet}: row count mismatch")
        values = table.to_pydict()
        frame = np.asarray(values["frame_index"], dtype=np.int64)
        timestamp = np.asarray(values["timestamp"], dtype=np.float32)
        global_indices = np.asarray(values["index"], dtype=np.int64)
        if not np.array_equal(frame, np.arange(length, dtype=np.int64)):
            raise DatasetValidationError(f"{parquet}: invalid frame_index")
        if not np.allclose(
            timestamp,
            frame.astype(np.float32) / np.float32(PUBLISHED_FPS),
            rtol=0,
            atol=1e-6,
        ):
            raise DatasetValidationError(f"{parquet}: invalid timestamp")
        if not np.array_equal(global_indices, np.arange(next_global_index, next_global_index + length)):
            raise DatasetValidationError(f"{parquet}: invalid global index")
        if set(values["episode_index"]) != {episode_index}:
            raise DatasetValidationError(f"{parquet}: invalid episode_index")
        if not set(values["task_index"]).issubset(task_ids):
            raise DatasetValidationError(f"{parquet}: unknown task_index")
        for key in ("observation.state", "observation.projected_gravity", "action.wbc", "action.motion_token"):
            array = np.asarray(values[key], dtype=np.float64)
            if not np.isfinite(array).all():
                raise DatasetValidationError(f"{parquet}: {key} has NaN/Inf")
        token = np.asarray(values["action.motion_token"], dtype=np.float64)
        if not np.allclose(token * 16, np.rint(token * 16), rtol=0, atol=1e-8):
            raise DatasetValidationError(f"{parquet}: token is off FSQ grid")
        for video_key in CAMERA_MAP.values():
            video = root / "videos" / f"chunk-{chunk:03d}" / video_key / f"episode_{episode_index:06d}.mp4"
            _probe_video(video, length)
            image_dir = root / "images" / video_key / f"episode_{episode_index:06d}"
            files = sorted(image_dir.glob("frame_*.jpg"))
            expected_names = [f"frame_{index:06d}.jpg" for index in range(length)]
            if [path.name for path in files] != expected_names:
                raise DatasetValidationError(f"{image_dir}: JPEG sequence mismatch")
            indices = range(length) if decode_all_images else (0, length // 2, length - 1)
            for index in indices:
                image = cv2.imread(str(image_dir / f"frame_{index:06d}.jpg"), cv2.IMREAD_COLOR)
                if image is None or image.shape != (480, 640, 3):
                    raise DatasetValidationError(f"{image_dir}: invalid JPEG {index}")
        next_global_index += length
        total_frames += length
    if total_frames != int(info.get("total_frames", -1)):
        raise DatasetValidationError("total_frames mismatch")
    if int(info.get("total_videos", -1)) != len(episodes) * 3:
        raise DatasetValidationError("total_videos mismatch")
    return {"episodes": len(episodes), "frames": total_frames, "fps": PUBLISHED_FPS}


def main() -> int:
    parser = argparse.ArgumentParser(description="validate strict ELF3 LeRobot v2.1")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--decode-all-images", action="store_true")
    args = parser.parse_args()
    try:
        report = validate_dataset(args.dataset, decode_all_images=args.decode_all_images)
    except (DatasetValidationError, OSError, ValueError, KeyError) as exc:
        print(f"DATASET ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"PASS: episodes={report['episodes']} frames={report['frames']} fps={report['fps']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
