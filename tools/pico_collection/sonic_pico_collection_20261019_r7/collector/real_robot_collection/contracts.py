from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence

import numpy as np


BODY_JOINT_NAMES = (
    "waist_y_joint", "waist_x_joint", "waist_z_joint",
    "l_hip_y_joint", "l_hip_x_joint", "l_hip_z_joint",
    "l_knee_y_joint", "l_ankle_y_joint", "l_ankle_x_joint",
    "r_hip_y_joint", "r_hip_x_joint", "r_hip_z_joint",
    "r_knee_y_joint", "r_ankle_y_joint", "r_ankle_x_joint",
    "l_shoulder_y_joint", "l_shoulder_x_joint", "l_shoulder_z_joint",
    "l_elbow_y_joint", "l_wrist_x_joint", "l_wrist_y_joint",
    "l_wrist_z_joint", "r_shoulder_y_joint", "r_shoulder_x_joint",
    "r_shoulder_z_joint", "r_elbow_y_joint", "r_wrist_x_joint",
    "r_wrist_y_joint", "r_wrist_z_joint",
)

HEAD_JOINT_NAMES = ("head_z_joint", "head_y_joint")
CANONICAL_JOINT_NAMES = BODY_JOINT_NAMES + HEAD_JOINT_NAMES

SONIC_DEFAULT_POSITION = np.asarray(
    (
        0.0, 0.0, 0.0,
        -0.3, 0.0, 0.0, 0.6, -0.3, 0.0,
        -0.3, 0.0, 0.0, 0.6, -0.3, 0.0,
        0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
        0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
    ),
    dtype=np.float32,
)

SONIC_ACTION_SCALE = np.asarray(
    (
        0.230525229, 0.153683486, 0.141706486,
        0.141706486, 0.141706486, 0.230525229,
        0.212559729, 0.373212313, 0.229663314,
        0.141706486, 0.141706486, 0.230525229,
        0.212559729, 0.373212313, 0.229663314,
        0.230525229, 0.230525229, 0.373201170,
        0.230525229, 0.373201170, 0.373201170, 0.373201170,
        0.230525229, 0.230525229, 0.373201170,
        0.230525229, 0.373201170, 0.373201170, 0.373201170,
    ),
    dtype=np.float32,
)

DIAGNOSTIC_ACTION_DIM = len(BODY_JOINT_NAMES)
DIAGNOSTIC_ACTION_HISTORY = 10

# Published LeRobot v2.1 contract.  The 29-D value reconstructed from
# ActuatorCmds below is useful diagnostic context, but it is deliberately not
# called a motion token: the training feature is a different, 64-D post-FSQ
# latent produced by the customer ELF3 Sonic step28800 encoder.
PUBLISHED_FPS = 30
STATE_DIM = 31
WBC_ACTION_DIM = 31
MOTION_TOKEN_DIM = 64
GRIPPER_STATE_JOINT_NAMES = (
    "left_gripper",
    "right_gripper",
)
GRIPPER_COMMAND_NAMES = ("left_gripper", "right_gripper")


def finite_vector(value, size: int, label: str, dtype=np.float32) -> np.ndarray:
    result = np.asarray(value, dtype=dtype).reshape(-1)
    if result.shape != (size,):
        raise ValueError(f"{label} must have shape ({size},), got {result.shape}")
    if not np.isfinite(result).all():
        raise ValueError(f"{label} contains NaN or Inf")
    return np.ascontiguousarray(result)


def validate_names(names: Iterable[str], values_size: int, label: str) -> tuple[str, ...]:
    result = tuple(str(name) for name in names)
    if len(result) != int(values_size):
        raise ValueError(f"{label}: names={len(result)} values={values_size}")
    if not all(result):
        raise ValueError(f"{label}: empty joint name")
    if len(set(result)) != len(result):
        raise ValueError(f"{label}: duplicate joint names")
    return result


def reorder_named(
    values,
    incoming_names: Iterable[str],
    expected_names: Iterable[str],
    label: str,
) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float32).reshape(-1)
    names = validate_names(incoming_names, vector.size, label)
    index = {name: position for position, name in enumerate(names)}
    expected = tuple(expected_names)
    missing = [name for name in expected if name not in index]
    if missing:
        raise ValueError(f"{label}: missing joints {missing}")
    result = np.asarray([vector[index[name]] for name in expected], dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError(f"{label}: contains NaN or Inf")
    return np.ascontiguousarray(result)


def canonicalize_named(
    values,
    incoming_names: Iterable[str],
    *,
    expected_names: Sequence[str] = CANONICAL_JOINT_NAMES,
    required_names: Sequence[str] = BODY_JOINT_NAMES,
    label: str,
    fill_value: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Map a named vector to a fixed layout and return an explicit presence mask."""

    vector = np.asarray(values, dtype=np.float32).reshape(-1)
    names = validate_names(incoming_names, vector.size, label)
    if not np.isfinite(vector).all():
        raise ValueError(f"{label}: contains NaN or Inf")
    index = {name: position for position, name in enumerate(names)}
    missing_required = [name for name in required_names if name not in index]
    if missing_required:
        raise ValueError(f"{label}: missing required joints {missing_required}")
    output = np.full(len(expected_names), float(fill_value), dtype=np.float32)
    present = np.zeros(len(expected_names), dtype=bool)
    for target, name in enumerate(expected_names):
        source = index.get(name)
        if source is not None:
            output[target] = vector[source]
            present[target] = True
    return output, present


def canonicalize_optional_named(
    values,
    incoming_names: Iterable[str],
    *,
    present_names: Sequence[str],
    label: str,
) -> tuple[np.ndarray, np.ndarray]:
    vector = np.asarray(values, dtype=np.float32).reshape(-1)
    if vector.size == 0:
        return (
            np.zeros(len(CANONICAL_JOINT_NAMES), dtype=np.float32),
            np.zeros(len(CANONICAL_JOINT_NAMES), dtype=bool),
        )
    return canonicalize_named(
        vector,
        incoming_names,
        expected_names=CANONICAL_JOINT_NAMES,
        required_names=present_names,
        label=label,
    )


def effective_continuous_action_29(command_position, command_names: Iterable[str]) -> np.ndarray:
    target = reorder_named(
        command_position,
        command_names,
        BODY_JOINT_NAMES,
        "SONIC actuator command",
    )
    token = (target - SONIC_DEFAULT_POSITION) / SONIC_ACTION_SCALE
    if not np.isfinite(token).all():
        raise ValueError("effective action token contains NaN or Inf")
    return np.ascontiguousarray(token, dtype=np.float32)


def validate_effective_continuous_action_29(value) -> np.ndarray:
    return finite_vector(value, DIAGNOSTIC_ACTION_DIM, "effective_continuous_action_29")


def pack_state_31(body_29, gripper_state_2) -> np.ndarray:
    """Pack measured body and gripper feedback in the simulation-baseline order."""

    body = finite_vector(body_29, len(BODY_JOINT_NAMES), "body_state", dtype=np.float64)
    gripper = finite_vector(
        gripper_state_2,
        len(GRIPPER_STATE_JOINT_NAMES),
        "gripper_state",
        dtype=np.float64,
    )
    return np.ascontiguousarray(np.concatenate((body, gripper)), dtype=np.float64)


def pack_action_31(body_29, gripper_command_2) -> np.ndarray:
    """Pack body29 plus one independent coordinate per physical gripper."""

    body = finite_vector(body_29, len(BODY_JOINT_NAMES), "body_action", dtype=np.float64)
    command = finite_vector(
        gripper_command_2,
        len(GRIPPER_COMMAND_NAMES),
        "gripper_command",
        dtype=np.float64,
    )
    left, right = command
    return np.ascontiguousarray(
        np.concatenate((body, [left, right])), dtype=np.float64
    )


def validate_motion_token(value, *, require_fsq_grid: bool = True) -> np.ndarray:
    """Validate, never synthesize, an ELF3 step28800 post-FSQ token."""

    token = finite_vector(value, MOTION_TOKEN_DIM, "action.motion_token", dtype=np.float64)
    if require_fsq_grid:
        units = token * 16.0
        if not np.allclose(units, np.rint(units), rtol=0.0, atol=1.0e-8):
            raise ValueError("action.motion_token is not on the expected 1/16 FSQ grid")
        if token.min() < -1.0 - 1.0e-8 or token.max() > 0.9375 + 1.0e-8:
            raise ValueError("action.motion_token is outside the expected FSQ range")
    return token


def projected_gravity_wxyz(quaternion) -> np.ndarray:
    quat = finite_vector(quaternion, 4, "base_quaternion_wxyz", dtype=np.float64)
    norm = float(np.linalg.norm(quat))
    if norm <= 1.0e-12:
        raise ValueError("base quaternion has zero norm")
    w, x, y, z = quat / norm
    projected = np.asarray(
        [
            2.0 * (w * y - x * z),
            -2.0 * (y * z + w * x),
            -1.0 + 2.0 * (x * x + y * y),
        ],
        dtype=np.float32,
    )
    projected /= max(float(np.linalg.norm(projected)), 1.0e-12)
    return projected


def _quat_multiply_wxyz(left, right) -> np.ndarray:
    aw, ax, ay, az = finite_vector(left, 4, "left quaternion", dtype=np.float64)
    bw, bx, by, bz = finite_vector(right, 4, "right quaternion", dtype=np.float64)
    value = np.asarray(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=np.float64,
    )
    norm = float(np.linalg.norm(value))
    if norm <= 1.0e-12:
        raise ValueError("quaternion product has zero norm")
    return value / norm


def _axis_angle_quaternion_wxyz(axis: str, angle: float) -> np.ndarray:
    half = 0.5 * float(angle)
    c, s = math.cos(half), math.sin(half)
    if axis == "x":
        return np.asarray([c, s, 0.0, 0.0], dtype=np.float64)
    if axis == "y":
        return np.asarray([c, 0.0, s, 0.0], dtype=np.float64)
    if axis == "z":
        return np.asarray([c, 0.0, 0.0, s], dtype=np.float64)
    raise ValueError(f"unsupported quaternion axis {axis!r}")


def sonic_root_quaternion_wxyz(torso_quaternion_wxyz, body_joint_position) -> np.ndarray:
    torso = finite_vector(
        torso_quaternion_wxyz, 4, "torso_quaternion_wxyz", dtype=np.float64
    )
    torso_norm = float(np.linalg.norm(torso))
    if torso_norm <= 1.0e-12:
        raise ValueError("torso quaternion has zero norm")
    q = torso / torso_norm
    joints = finite_vector(body_joint_position, 29, "body_joint_position")
    q = _quat_multiply_wxyz(q, _axis_angle_quaternion_wxyz("y", joints[0]))
    q = _quat_multiply_wxyz(q, _axis_angle_quaternion_wxyz("x", joints[1]))
    return _quat_multiply_wxyz(q, _axis_angle_quaternion_wxyz("z", joints[2]))


def sonic_projected_gravity(torso_quaternion_wxyz, body_joint_position) -> np.ndarray:
    return projected_gravity_wxyz(
        sonic_root_quaternion_wxyz(torso_quaternion_wxyz, body_joint_position)
    )


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    instruction_text: str


def tasks_from_config(dataset_config: dict) -> tuple[TaskSpec, ...]:
    raw_tasks = dataset_config.get("tasks")
    if not isinstance(raw_tasks, list) or not raw_tasks:
        raise ValueError("dataset.tasks must be a non-empty list")
    tasks: list[TaskSpec] = []
    for index, item in enumerate(raw_tasks):
        if not isinstance(item, dict):
            raise ValueError(f"dataset.tasks[{index}] must be a mapping")
        task_id = str(item.get("id", "")).strip()
        instruction = str(item.get("instruction", "")).strip()
        if not task_id or not instruction:
            raise ValueError(f"dataset.tasks[{index}] requires id and instruction")
        tasks.append(TaskSpec(task_id, instruction))
    ids = [task.task_id for task in tasks]
    if len(set(ids)) != len(ids):
        raise ValueError("dataset.tasks contains duplicate ids")
    return tuple(tasks)
