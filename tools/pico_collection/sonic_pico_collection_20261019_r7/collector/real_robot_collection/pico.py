from __future__ import annotations

from collections import deque
import json
import threading
import time
from dataclasses import dataclass

import numpy as np

try:
    import zmq
except ImportError:  # Pure schema/tests do not need a live socket.
    zmq = None


HEADER_SIZE = 1280
DTYPES = {
    "f32": np.dtype("<f4"),
    "f64": np.dtype("<f8"),
    "i32": np.dtype("<i4"),
    "i64": np.dtype("<i8"),
    "u8": np.dtype("u1"),
    "bool": np.dtype("?"),
}

RAW_REQUIRED_FIELDS = (
    "smpl_pose",
    "smpl_joints",
    "body_quat_w",
    "joint_pos",
    "joint_vel",
    "vr_position",
    "vr_orientation",
    "frame_index",
    "left_trigger",
    "right_trigger",
    "left_grip",
    "right_grip",
    "pico_dt",
    "pico_fps",
    "timestamp_realtime",
    "timestamp_monotonic",
    "left_hand_joints",
    "right_hand_joints",
    "heading_increment",
    "stream_mode",
    "calibration_ready",
)

SMPL_REQUIRED_FIELDS = (
    "term1_local",
    "root_quat",
    "wrist",
    "frame_index",
    "source_ready",
    "source_stream_mode",
    "source_calibration_ready",
)


def smpl_reference_ready(snapshot):
    if snapshot is None:
        return False
    fields = snapshot.fields
    try:
        if scalar(fields, 'source_chunk', 0) == 1:
            indices = np.asarray(fields['frame_index']).reshape(-1)
            count = len(indices)
            return (count >= 10 and np.issubdtype(indices.dtype, np.integer)
                    and bool(np.all(np.diff(indices) == 1))
                    and scalar(fields, 'source_stream_epoch', 0) > 0
                    and scalar(fields, 'source_received_monotonic_ns', 0) > 0
                    and scalar(fields, 'valid_horizon', 0) >= 10
                    and scalar(fields, 'clamp_slots', -1) == 0
                    and all(np.asarray(fields[name]).shape == (count, width)
                            and bool(np.isfinite(fields[name]).all())
                            for name, width in (('term1_local',72),('root_quat',4),('wrist',6))))
        return (scalar(fields, 'source_ready', False) == 1
                and scalar(fields, 'source_stream_mode', -1) == 1
                and scalar(fields, 'source_calibration_ready', False) == 1)
    except (KeyError, IndexError, TypeError, ValueError, OverflowError):
        return False


def decode_packed_message(message: bytes, topic: str) -> dict[str, np.ndarray] | None:
    prefix = topic.encode("utf-8")
    if not message.startswith(prefix):
        return None
    payload = memoryview(message)[len(prefix):]
    if len(payload) < HEADER_SIZE:
        raise ValueError("packed message is shorter than the 1280-byte header")
    raw_header = bytes(payload[:HEADER_SIZE]).split(b"\x00", 1)[0]
    if not raw_header:
        raise ValueError("packed message has an empty header")
    header = json.loads(raw_header.decode("utf-8"))
    if header.get("endian", "le") != "le":
        raise ValueError(f"unsupported packed-message endian {header.get('endian')!r}")
    raw_fields = header.get("fields")
    if not isinstance(raw_fields, list):
        raise ValueError("packed-message header.fields must be a list")
    binary = payload[HEADER_SIZE:]
    offset = 0
    result: dict[str, np.ndarray] = {}
    for field in raw_fields:
        if not isinstance(field, dict):
            raise ValueError("packed-message field descriptor must be a mapping")
        name = str(field["name"])
        if not name or name in result:
            raise ValueError(f"duplicate or empty packed field name {name!r}")
        dtype = DTYPES.get(str(field["dtype"]))
        if dtype is None:
            raise ValueError(f"unsupported packed dtype for {name}: {field['dtype']}")
        shape = tuple(int(value) for value in field.get("shape", []))
        if any(value < 0 for value in shape):
            raise ValueError(f"negative shape for packed field {name}: {shape}")
        count = int(np.prod(shape, dtype=np.int64)) if shape else 1
        size = count * dtype.itemsize
        if offset + size > len(binary):
            raise ValueError(f"packed field {name} exceeds payload")
        values = np.frombuffer(binary[offset:offset + size], dtype=dtype, count=count)
        result[name] = values.reshape(shape).copy()
        offset += size
    if offset != len(binary):
        raise ValueError(f"packed message has {len(binary) - offset} trailing bytes")
    return result


def scalar(fields: dict[str, np.ndarray] | None, key: str, default=0.0):
    if not fields or key not in fields:
        return default
    values = np.asarray(fields[key]).reshape(-1)
    return values[-1].item() if values.size else default


def _json_safe(fields: dict[str, np.ndarray]) -> dict:
    return {key: np.asarray(value).tolist() for key, value in fields.items()}


@dataclass(frozen=True)
class PackedSnapshot:
    fields: dict[str, np.ndarray]
    received_monotonic: float

    def json_safe_fields(self) -> dict:
        return _json_safe(self.fields)

    @property
    def frame_index(self) -> int | None:
        if "frame_index" not in self.fields:
            return None
        values = np.asarray(self.fields["frame_index"]).reshape(-1)
        return int(values[-1]) if values.size else None


@dataclass(frozen=True)
class PicoSnapshot(PackedSnapshot):
    @property
    def toggle_collection(self) -> bool:
        return bool(scalar(self.fields, "toggle_data_collection", False))

    @property
    def abort(self) -> bool:
        return bool(scalar(self.fields, "toggle_data_abort", False))

    @property
    def calibrated(self) -> bool:
        return bool(scalar(self.fields, "calibration_ready", False))

    @property
    def stream_mode(self) -> int:
        return int(scalar(self.fields, "stream_mode", -1))

    @property
    def triggers(self) -> tuple[float, float] | None:
        if "left_trigger" not in self.fields or "right_trigger" not in self.fields:
            return None
        left = float(np.clip(scalar(self.fields, "left_trigger"), 0.0, 1.0))
        right = float(np.clip(scalar(self.fields, "right_trigger"), 0.0, 1.0))
        return left, right


class PackedSubscriber:
    def __init__(
        self,
        host: str,
        port: int,
        topic: str,
        *,
        snapshot_type=PackedSnapshot,
        history_size: int = 256,
        required_fields: tuple[str, ...] = (),
    ) -> None:
        self.host = str(host)
        self.port = int(port)
        self.topic = str(topic)
        self.snapshot_type = snapshot_type
        self.required_fields = tuple(required_fields)
        self._lock = threading.Lock()
        self._history: deque[PackedSnapshot] = deque(maxlen=int(history_size))
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._error: str | None = None
        self._received = 0

    def start(self) -> None:
        if zmq is None:
            raise RuntimeError("pyzmq is required for SONIC packed-stream input")
        if self._thread is not None:
            raise RuntimeError("packed subscriber is already started")
        self._thread = threading.Thread(
            target=self._run,
            name=f"zmq-{self.topic}",
            daemon=True,
        )
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def latest(self) -> PackedSnapshot | None:
        with self._lock:
            return self._history[-1] if self._history else None

    def nearest(self, target_monotonic: float, max_delta_s: float) -> PackedSnapshot | None:
        with self._lock:
            history = tuple(self._history)
        if not history:
            return None
        sample = min(
            history,
            key=lambda item: abs(item.received_monotonic - target_monotonic),
        )
        if abs(sample.received_monotonic - target_monotonic) > float(max_delta_s):
            return None
        return sample

    def latest_at_or_before(
        self, target_monotonic: float, max_age_s: float
    ) -> PackedSnapshot | None:
        """Return causal data only; future packets must not label an earlier row."""

        with self._lock:
            history = tuple(
                item
                for item in self._history
                if item.received_monotonic <= float(target_monotonic)
            )
        if not history:
            return None
        sample = max(history, key=lambda item: item.received_monotonic)
        if float(target_monotonic) - sample.received_monotonic > float(max_age_s):
            return None
        return sample

    def after(self, received_monotonic: float) -> tuple[PackedSnapshot, ...]:
        """Return buffered packets not yet consumed by an event reader."""

        with self._lock:
            return tuple(
                item
                for item in self._history
                if item.received_monotonic > float(received_monotonic)
            )

    @property
    def error(self) -> str | None:
        return self._error

    @property
    def received(self) -> int:
        return self._received

    def _run(self) -> None:
        context = zmq.Context()
        socket = context.socket(zmq.SUB)
        socket.setsockopt(zmq.RCVHWM, 4)
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt_string(zmq.SUBSCRIBE, self.topic)
        socket.connect(f"tcp://{self.host}:{self.port}")
        poller = zmq.Poller()
        poller.register(socket, zmq.POLLIN)
        try:
            while not self._stop.is_set():
                if socket not in dict(poller.poll(100)):
                    continue
                try:
                    fields = decode_packed_message(socket.recv(), self.topic)
                    if fields is None:
                        continue
                    missing = [name for name in self.required_fields if name not in fields]
                    if missing:
                        raise ValueError(f"{self.topic} packet missing fields {missing}")
                    snapshot = self.snapshot_type(fields, time.monotonic())
                    with self._lock:
                        self._history.append(snapshot)
                    self._received += 1
                    self._error = None
                except Exception as exc:
                    self._error = str(exc)
        finally:
            socket.close(0)
            context.term()


class PicoSubscriber(PackedSubscriber):
    def __init__(self, host: str, port: int, topic: str, *, history_size: int = 256):
        super().__init__(
            host,
            port,
            topic,
            snapshot_type=PicoSnapshot,
            history_size=history_size,
            required_fields=RAW_REQUIRED_FIELDS,
        )


def smpl_subscriber(
    host: str, port: int, topic: str, *, history_size: int = 256
) -> PackedSubscriber:
    return PackedSubscriber(
        host,
        port,
        topic,
        history_size=history_size,
        # Readiness validation accepts legacy flags OR complete source chunks.
        required_fields=SMPL_REQUIRED_FIELDS[:4],
    )
