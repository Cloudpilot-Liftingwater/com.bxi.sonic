from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, TypeVar

import cv2
import numpy as np


@dataclass(frozen=True)
class TimedSample:
    received_monotonic: float
    stamp_unix: float


@dataclass(frozen=True)
class StateSample(TimedSample):
    sequence: int
    position: np.ndarray
    velocity: np.ndarray
    effort: np.ndarray
    driver_temperature: np.ndarray
    motor_temperature: np.ndarray
    joint_present: np.ndarray
    driver_temperature_present: np.ndarray
    motor_temperature_present: np.ndarray


@dataclass(frozen=True)
class ImuSample(TimedSample):
    quaternion_wxyz: np.ndarray
    angular_velocity: np.ndarray
    linear_acceleration: np.ndarray
    frame_id: str


@dataclass(frozen=True)
class GripperSample(TimedSample):
    values: np.ndarray
    raw: dict | None = None


@dataclass(frozen=True)
class ImageSample(TimedSample):
    sequence: int
    data: np.ndarray
    encoding: str
    frame_id: str


@dataclass(frozen=True)
class CommandSample(TimedSample):
    sequence: int
    position: np.ndarray
    velocity: np.ndarray
    torque: np.ndarray
    kp: np.ndarray
    kd: np.ndarray
    joint_present: np.ndarray
    diagnostic_action: np.ndarray
    diagnostic_action_history: np.ndarray | None


@dataclass(frozen=True)
class SonicInferenceSample(TimedSample):
    inference_sequence: int
    model_id: str
    model_sha256: str
    target_position: np.ndarray
    motion_token: np.ndarray
    policy_status: str


@dataclass(frozen=True)
class JsonSample(TimedSample):
    value: dict


class SequenceResolver:
    """Prefer an explicit numeric frame id, with a deterministic local fallback."""

    MODES = ("frame_id", "counter", "frame_id_or_counter")

    def __init__(self, mode: str, prefix: str, counter_start: int = -1) -> None:
        if mode not in self.MODES:
            raise ValueError(f"sequence.mode must be one of {self.MODES}, got {mode!r}")
        self.mode = mode
        self.prefix = str(prefix)
        self.counter = int(counter_start)

    def parse(self, frame_id: str) -> int | None:
        text = str(frame_id)
        if not text.startswith(self.prefix):
            return None
        try:
            sequence = int(text[len(self.prefix):])
        except ValueError:
            return None
        return sequence if sequence >= 0 else None

    def sequence(self, frame_id: str) -> int:
        explicit = self.parse(frame_id)
        if explicit is not None and self.mode != "counter":
            self.counter = max(self.counter, explicit)
            return explicit
        if self.mode == "frame_id":
            raise ValueError(
                f"frame_id={frame_id!r} does not match "
                f"{self.prefix}<nonnegative integer>"
            )
        self.counter += 1
        return self.counter

    # Backward-compatible spelling used by older deployments/tests.
    state_sequence = sequence


T = TypeVar("T", bound=TimedSample)


def latest_at_or_before(
    samples: Iterable[T], target_monotonic: float, max_age_s: float
) -> T | None:
    candidates = [item for item in samples if item.received_monotonic <= target_monotonic]
    if not candidates:
        return None
    sample = max(candidates, key=lambda item: item.received_monotonic)
    if target_monotonic - sample.received_monotonic > float(max_age_s):
        return None
    return sample


def matching_sonic_inference(
    samples: Iterable[SonicInferenceSample],
    command: CommandSample,
    target_monotonic: float,
    max_age_s: float,
    action_atol: float,
) -> SonicInferenceSample | None:
    """Return the newest atomic action/token bundle matching a motor command."""

    body_size = 29
    candidates = [
        item
        for item in samples
        if item.received_monotonic <= target_monotonic
        and abs(item.received_monotonic - command.received_monotonic) <= max_age_s
        and np.allclose(
            item.target_position[:body_size],
            command.position[:body_size],
            rtol=0.0,
            atol=action_atol,
        )
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda item: (item.inference_sequence, item.received_monotonic))


def decode_compressed_image(data: bytes) -> np.ndarray:
    encoded = np.frombuffer(data, dtype=np.uint8)
    bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("OpenCV could not decode CompressedImage")
    return np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def decode_raw_image(msg) -> np.ndarray:
    encoding = str(msg.encoding).lower()
    channels = {
        "rgb8": 3,
        "bgr8": 3,
        "rgba8": 4,
        "bgra8": 4,
        "mono8": 1,
    }.get(encoding)
    if channels is None:
        raise ValueError(f"unsupported raw RGB image encoding {msg.encoding!r}")
    height, width, step = int(msg.height), int(msg.width), int(msg.step)
    if height <= 0 or width <= 0 or step < width * channels:
        raise ValueError(
            f"malformed Image dimensions height={height} width={width} step={step}"
        )
    flat = np.frombuffer(msg.data, dtype=np.uint8)
    if flat.size < height * step:
        raise ValueError(f"Image data has {flat.size} bytes, expected {height * step}")
    rows = flat[: height * step].reshape(height, step)
    image = rows[:, : width * channels].reshape(height, width, channels)
    if encoding == "rgb8":
        rgb = image
    elif encoding == "bgr8":
        rgb = image[:, :, ::-1]
    elif encoding == "rgba8":
        rgb = image[:, :, :3]
    elif encoding == "bgra8":
        rgb = image[:, :, [2, 1, 0]]
    else:
        rgb = np.repeat(image, 3, axis=2)
    return np.ascontiguousarray(rgb)


def decode_depth_image(msg) -> tuple[np.ndarray, str]:
    encoding = str(msg.encoding).upper()
    if encoding in ("16UC1", "MONO16"):
        dtype = np.dtype(">u2" if bool(msg.is_bigendian) else "<u2")
    elif encoding == "32FC1":
        dtype = np.dtype(">f4" if bool(msg.is_bigendian) else "<f4")
    else:
        raise ValueError(f"unsupported depth image encoding {msg.encoding!r}")
    height, width, step = int(msg.height), int(msg.width), int(msg.step)
    minimum_step = width * dtype.itemsize
    if height <= 0 or width <= 0 or step < minimum_step:
        raise ValueError(
            f"malformed depth dimensions height={height} width={width} step={step}"
        )
    raw = memoryview(msg.data)
    if len(raw) < height * step:
        raise ValueError(f"depth data has {len(raw)} bytes, expected {height * step}")
    output = np.empty((height, width), dtype=dtype.newbyteorder("="))
    for row in range(height):
        begin = row * step
        output[row] = np.frombuffer(
            raw[begin:begin + minimum_step], dtype=dtype, count=width
        ).astype(output.dtype, copy=False)
    if np.issubdtype(output.dtype, np.floating) and not np.isfinite(output).all():
        raise ValueError("depth image contains NaN or Inf")
    return np.ascontiguousarray(output), encoding


def quaternion_multiply_wxyz(left, right) -> np.ndarray:
    aw, ax, ay, az = np.asarray(left, dtype=np.float64).reshape(4)
    bw, bx, by, bz = np.asarray(right, dtype=np.float64).reshape(4)
    result = np.asarray(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=np.float64,
    )
    norm = float(np.linalg.norm(result))
    if norm <= 1.0e-12:
        raise ValueError("quaternion product has zero norm")
    return result / norm
