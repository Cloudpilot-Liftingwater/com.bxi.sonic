"""ELF3 field adapter for the vendored NVIDIA SONIC/LeRobot writer."""
from pathlib import Path

from contract import CONTRACT_ID, COORDINATE, POSITION_UNIT, CAMERAS, FPS, SCHEMA, VECTOR_NAMES, MODEL_SHA, modality


def dependencies():
    """Fail before producing output; never install packages or start hardware."""
    try:
        from vendor.sonic_data.exporter import Gr00tDataExporter
    except (ImportError, RuntimeError) as exc:
        raise RuntimeError(
            "Official SONIC writer dependencies unavailable. Use the isolated export "
            "environment described in tools/gr00t_dataset/README.md; do not alter ROS."
        ) from exc
    return Gr00tDataExporter


def create(path, task, *, camera_shapes=None, online=False, record_smpl=False, source_metadata=None):
    features = {k: dict(dtype="float32", shape=(len(names),), names=names)
                for k, names in VECTOR_NAMES.items()}
    if record_smpl:
        features["teleop.smpl_pose"] = dict(dtype="float32", shape=(63,), names=[f"smpl_{i}" for i in range(63)])
    features.update({
        "observation.images." + name: dict(dtype="video", shape=(camera_shapes or {}).get(name, (480, 640, 3)),
                                          names=["height", "width", "channels"])
        for name in CAMERAS
    })
    exporter = dependencies().create(
        save_root=Path(path), fps=FPS, features=features, modality_config=modality(),
        task=task, robot_type=SCHEMA, overwrite_existing=False,
        script_config=dict(
            **(source_metadata or {}), tokenizer_sha256=MODEL_SHA,
            adapter="elf3_ros_official_online_v1" if online else "elf3_ros_saved_source_v1", gripper_coordinate=COORDINATE, gripper_position_unit=POSITION_UNIT, contract_id=CONTRACT_ID,
            source="ELF3 ROS2 synchronized actual inference rows",
            camera_policy="latest_available_rgb_on_actual_inference_rows",
        ),
    )
    if online:
        for writer in exporter.video_writers.values():
            writer.encoder_options = {"preset": "ultrafast", "crf": "18"}
            # A single 1080p encoder was the throughput bottleneck. Bound the
            # HD codec to four threads; small wrist streams retain one each.
            writer.thread_count = 4 if writer.width * writer.height >= 1920 * 1080 else 1
    return exporter


def close(exporter):
    """Attempt to close every camera even if one encoder has already failed."""
    errors = []
    for writer in exporter.video_writers.values():
        try:
            writer.stop()
        except BaseException as exc:
            errors.append(exc)
    if errors:
        raise errors[0]


def verify_videos(exporter):
    """Read back final frames, video dimensions and frame rate."""
    import av
    decoded = 0
    for index, episode in exporter.meta.episodes.items():
        for key in exporter.video_keys:
            path = exporter.root / exporter.meta.get_video_file_path(index, key)
            with av.open(str(path)) as container:
                stream = container.streams.video[0]
                if (stream.width, stream.height, float(stream.average_rate)) != (exporter.features[key]["shape"][1], exporter.features[key]["shape"][0], FPS):
                    raise ValueError(f"Unexpected video dimensions/rate: {path}")
                n = sum(1 for _ in container.decode(video=0))
                if n != episode["length"]:
                    raise ValueError(f"Video incomplete: {path}: {n}/{episode['length']}")
                decoded += n
    return decoded
