#!/usr/bin/env python3
"""ELF3 ROS recordings -> official SONIC LeRobot writer. Offline; never opens ROS/CAN."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile

import cv2
import numpy as np

from contract import CAMERAS, FPS, HORIZON, ISAAC_COMMIT, JOINTS, MODEL_SHA
from contract import ABI, RAW_JOINTS, SCHEMA, VECTOR_NAMES, modality
import official_writer
from vendor.gr00t_stats import generate_stats

REPO = Path(__file__).resolve().parents[2]
R7 = REPO / "tools/pico_collection/sonic_pico_collection_20261019_r7"


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def preflight(task):
    if not isinstance(task, str) or len(task.strip()) < 3:
        raise ValueError("Set training_task to the actual task instruction before recording.")
    if task.lower().startswith("diagnostic") or task.strip().lower() in ("test", "demo"):
        raise ValueError("A diagnostic placeholder is not a task instruction.")
    official_writer.dependencies()


def vector(value, length, label):
    a = np.asarray(value, dtype=np.float32)
    if a.shape != (length,) or not np.isfinite(a).all():
        raise ValueError(f"Invalid {label}: expected {length} finite values")
    return a


def inspect_source(episode):
    """Read raw motor angles; export body29 + measured/commanded grippers, never head."""
    sys.path.insert(0, str(R7))
    from verify_saved import verify
    meta = json.loads((episode / "meta.json").read_text())
    if (meta.get("discarded") or meta.get("incomplete")
            or meta.get("excluded_from_training") or not meta["termination"]["success"]):
        raise ValueError("Discarded/incomplete/failed recording cannot be exported")
    if meta["joint_contract"]["canonical_names"] != RAW_JOINTS:
        raise ValueError("Unexpected raw joint order")
    if meta.get("fps") != FPS:
        raise ValueError(
            f"This project uses {FPS} Hz source rows (not a universal GR00T rule); "
            f"source is {meta.get('fps')!r}. Do not relabel or duplicate 30 Hz rows."
        )
    extra = meta["metadata"]
    if extra.get("camera_left_right_verified") is not True:
        raise ValueError("Physical camera roles have not been confirmed")
    if extra["motion_token"]["sha256"] != MODEL_SHA:
        raise ValueError("Different SONIC tokenizer; use a separate versioned training contract")
    if any(x["protocol"] != "bxi" for x in meta["gripper_identity_contract"]):
        raise ValueError("This training profile is for dual BXI; DM raw capture is unchanged")
    if meta.get("synchronization", {}).get("clock_jump", True):
        raise ValueError("Source clock jumped or its validity is unknown")
    audit = verify(episode, expected_ports=extra["physical_ports"])
    rows = []
    with (episode / "frames.jsonl").open() as stream:
        for line in stream:
            raw = json.loads(line)
            for key in ("observation.joint_present", "action.joint_present"):
                mask = raw[key]
                if len(mask) != 31 or any(v is not True for v in mask[:29]):
                    raise ValueError(f"row {raw['idx']}: missing body joint in {key}")
            grip = raw["gripper"]
            if grip.get("state_is_measured") is not True or grip.get("command_is_final") is not True:
                raise ValueError("Gripper labels must be measured feedback and final commands")
            for item in grip["native_raw_sample"]["samples"]:
                if (abs(float(item["closed_rad"])) > .005
                        or abs(float(item["open_rad"]) - np.deg2rad(300)) > .005):
                    raise ValueError("BXI coordinate differs from absolute closed0/open300 degrees")
            # verify() already checks units, ordered side/bus/ID identity,
            # freshness and equality of these vectors to the underlying motor samples.
            measured = vector(grip["state"], 2, "measured gripper motor radians")
            commanded = vector(grip["command"], 2, "commanded gripper motor radians")
            values = {
                "observation.state": list(raw["observation.state"][:29]) + measured.tolist(),
                "observation.projected_gravity": raw["observation.projected_gravity"],
                "action.motion_token": raw["action.motion_token"],
                "action.wbc": list(raw["action"][:29]) + commanded.tolist(),
            }
            rows.append(dict(
                values={k: vector(v, len(VECTOR_NAMES[k]), k) for k, v in values.items()},
                idx=raw["idx"], t=float(raw["auxiliary"]["anchor_stamp_unix"]),
                segment=raw["auxiliary"].get("capture_segment", 0),
                images=raw["observation.images"], inference=raw["sonic.inference"]["sequence"],
                camera_stamps={k:raw["source_timestamps_unix"][k] for k in CAMERAS.values()},
                camera_sequences=raw["auxiliary"]["selected_camera_sequences"],
            ))
    return meta, audit, rows


def episode_sequences(rows):
    """Keep one episode after removing gaps, as in official SONIC post-processing.

    Retain real source rows and clocks for provenance. The official writer assigns
    new frame indices/timestamps and writes the corresponding three video frames.
    """
    # Preserve every actual source row, including jitter and short recordings.
    if not rows:
        raise ValueError("Empty recording")
    return [[dict(r, grid_error_s=0.0) for r in rows]], []


def training_image(frame):
    """Preserve the full field of view; all three views share the model input size."""
    source_h, source_w = frame.shape[:2]
    scale = min(640 / source_w, 480 / source_h)
    width, height = round(source_w * scale), round(source_h * scale)
    resized = cv2.resize(frame, (width, height),
                         interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    canvas = np.zeros((480, 640, 3), dtype=np.uint8)
    x, y = (640 - width) // 2, (480 - height) // 2
    canvas[y:y + height, x:x + width] = resized
    return canvas


def export(episodes, output, task=None, *, inspection_only=False):
    episodes = [Path(p).expanduser().resolve(strict=True) for p in episodes]
    if not episodes:
        raise ValueError("At least one saved source episode is required")
    if len(set(episodes)) != len(episodes):
        raise ValueError("Repeated source episode")
    output = Path(output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    if any(output == p or p in output.parents for p in episodes):
        raise ValueError("Training output must be outside the raw episode")
    inspected = []
    for episode in episodes:
        meta, audit, rows = inspect_source(episode)
        instruction = task if task is not None else meta["task_name"]
        if inspection_only:
            # Keep the recorded label (including an explicit diagnostic label)
            # and every real row for visual inspection. Never call this train-ready.
            official_writer.dependencies()
            sequences, excluded = [rows], []
        else:
            preflight(instruction)
            sequences, excluded = episode_sequences(rows)
        inspected.append((episode, meta, audit, sequences, excluded, instruction.strip()))
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="." + output.name + ".inprogress.", dir=output.parent))
    temp = staging / "dataset"
    sources, provenance = [], []
    writer = None
    offset = 0
    try:
        writer = official_writer.create(temp, inspected[0][5])
        for episode, meta, audit, sequences, excluded, instruction in inspected:
            source_id = len(sources)
            sources.append(dict(
                logical_recording=episode.parent.parent.name, source_episode=str(episode),
                frames_sha256=digest(episode / "frames.jsonl"), meta_sha256=digest(episode / "meta.json"),
                readback=audit, excluded_rows=excluded, source_cameras=meta["camera_contract"],
                gripper_identities=meta["gripper_identity_contract"],
            ))
            for rows in sequences:
                index, n = writer.meta.total_episodes, len(rows)
                from image_io import EpisodeImages
                with EpisodeImages(episode) as image_reader:
                    for row in rows:
                        frame = {**row["values"], "task": instruction}
                        for key, raw_key in CAMERAS.items():
                            image = image_reader.read(row["images"][raw_key])
                            if image is None:
                                raise ValueError("Missing or unreadable source image")
                            # ROS/raw JPEG is BGR; upstream takes uint8 RGB.
                            frame["observation.images." + key] = cv2.cvtColor(
                                training_image(image), cv2.COLOR_BGR2RGB)
                        writer.add_frame(frame)
                writer.save_episode()
                provenance.append(dict(
                    episode_index=index, source_id=source_id, capture_segments=[r["segment"] for r in rows],
                    source_row_indices=[r["idx"] for r in rows],
                    source_unix_s=[r["t"] for r in rows],
                    inference_sequences=[r["inference"] for r in rows],
                    camera_source_unix_s={k:[r["camera_stamps"][k] for r in rows] for k in CAMERAS.values()},
                    camera_source_sequences={k:[r["camera_sequences"][k] for r in rows] for k in CAMERAS.values()},
                    maximum_grid_error_s=(None if inspection_only else max(r["grid_error_s"] for r in rows)),
                ))
                offset += n
        count = writer.meta.total_episodes
        official_writer.close(writer)
        decoded = official_writer.verify_videos(writer)
        # GR00T's existing native processor needs quantiles. LeRobot v2.1 itself
        # supplies info/tasks/episodes/episodes_stats via the upstream writer.
        generate_stats(temp)
        dump(temp / "meta/source_map.json", dict(sources=sources, sequences=provenance))
        dump(temp / "meta/elf3_contract.json", dict(
            schema=SCHEMA, joint_dimension=31, joint_names=JOINTS,
            state_model_dimension=34, action_model_dimension=66, action_horizon=HORIZON,
            embodiment_tag=ABI["embodiment_tag"], projector_slot=ABI["projector_slot"],
            fps=FPS, sonic_control_hz=50, tokenizer_sha256=MODEL_SHA,
            gripper_coordinate="absolute_motor_radians", gripper_position_unit="rad",
            gripper_reference_degrees={"closed": 0, "open": 300},
            action_order=["motion_token[64]", "left_gripper[1]", "right_gripper[1]"],
            local_gr00t_base_commit=ISAAC_COMMIT, interpolation=False,
            image_transform="aspect_preserving_letterbox_640x480_black_padding",
            pause_policy="one_logical_recording_one_training_episode_pause_removed",
            inspection_only=inspection_only,
            training_ready=not inspection_only and all(x[2]["timing_review"] == "PASS" for x in inspected),
            output_time_basis="retained_row_index_divided_by_50; source times in source_map",
            missing_source_rows_reconstructed=False,
            export_release="20261021-export-04-mp4-source",
        ))
        # The existing native build_processor entry requires this provenance file.
        dump(temp / "meta/native_1d_derivation.json", dict(
            format_version=1, source_kind="real_robot_raw_episode",
            gripper_coordinate="absolute_motor_radians", unit="rad",
            body_rule="raw body indices 0:29", gripper_rule="copy measured/final motor radians",
            source_map="source_map.json", contract="elf3_contract.json",
            legacy_assets_modified=False, simulated_coordinate_conversion=False,
        ))
        report = dict(
            status="TRAINING_DATASET_READY", dataset=str(output), frames=offset,
            training_sequences=count, logical_recordings=len(inspected),
            excluded_raw_rows=sum(len(x[4]) for x in inspected),
            official_loader_run=False, hardware_commands_sent=0,
            storage_backend="NVIDIA_SONIC_Gr00tDataExporter",
            export_release="20261021-export-04-mp4-source",
            decoded_video_frames=decoded,
        )
        if report["excluded_raw_rows"]:
            report["status"] = "TRAINING_DATASET_READY_WITH_EXCLUSIONS"
        report["inspection_only"] = inspection_only
        report["timing_review_required"] = any(x[2]["timing_review"] != "PASS" for x in inspected)
        report["training_ready"] = not inspection_only and not report["timing_review_required"]
        if inspection_only:
            report["status"] = "INSPECTION_EXPORT_ONLY"
        elif report["timing_review_required"]:
            report["status"] = "EXPORTED_TIMING_REVIEW_REQUIRED"
        dump(temp / "EXPORT_RESULT.json", report)
        temp.rename(output)
        staging.rmdir()
        return report
    except BaseException as exc:
        if writer is not None:
            try:
                official_writer.close(writer)
            except BaseException as close_error:
                dump(staging / "CLOSE_FAILED.json", dict(error=str(close_error)))
        dump(temp / "EXPORT_FAILED.json", dict(error=str(exc), training_ready=False))
        raise RuntimeError(f"Export failed; raw unchanged, partial output retained: {temp}: {exc}") from exc


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("episodes", type=Path, nargs="*", help="Finalized raw episode directories")
    p.add_argument("--output", type=Path)
    p.add_argument("--check-dependencies", action="store_true")
    p.add_argument("--task", help="Actual instruction; otherwise use each recording's task_name")
    p.add_argument("--inspection-only", action="store_true",
                   help="Export every finalized real row for viewing, retain the original task label; never training-ready")
    a = p.parse_args()
    if a.check_dependencies:
        official_writer.dependencies()
        print("OFFICIAL_SONIC_WRITER_IMPORT_PASS; no ROS/devices opened")
        return
    if not a.episodes or a.output is None:
        p.error("episodes and --output are required for export")
    print(json.dumps(export(a.episodes, a.output, a.task, inspection_only=a.inspection_only), ensure_ascii=False))


if __name__ == "__main__":
    main()
