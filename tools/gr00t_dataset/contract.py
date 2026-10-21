"""Use the existing local ELF3 native-1DoF ABI; no new training configuration."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ABI = json.loads((ROOT / "elf3_native_1d_abi.json").read_text())
SCHEMA = "elf3_sonic_native_1d"  # Existing native_1d/derive_dataset.py robot_type.
FPS = ABI["frequency_hz"]
HORIZON = ABI["action_horizon"]
ISAAC_COMMIT = "b9955401d50c92a29258732e3ad6ccd579f1bdc0"
MODEL_SHA = "50303783f72ca61674b54987cab20abd11d0ced25c7042062fb03ef6355cdfdf"
BODY_JOINTS = [
    "waist_y_joint", "waist_x_joint", "waist_z_joint",
    "l_hip_y_joint", "l_hip_x_joint", "l_hip_z_joint", "l_knee_y_joint",
    "l_ankle_y_joint", "l_ankle_x_joint",
    "r_hip_y_joint", "r_hip_x_joint", "r_hip_z_joint", "r_knee_y_joint",
    "r_ankle_y_joint", "r_ankle_x_joint",
    "l_shoulder_y_joint", "l_shoulder_x_joint", "l_shoulder_z_joint", "l_elbow_y_joint",
    "l_wrist_x_joint", "l_wrist_y_joint", "l_wrist_z_joint",
    "r_shoulder_y_joint", "r_shoulder_x_joint", "r_shoulder_z_joint", "r_elbow_y_joint",
    "r_wrist_x_joint", "r_wrist_y_joint", "r_wrist_z_joint",
]
# The r7 source schema retains optional head slots for backwards-compatible readback.
# They are neither required nor passed to the training dataset.
RAW_JOINTS = BODY_JOINTS + ["head_z_joint", "head_y_joint"]
JOINTS = BODY_JOINTS + ["l_gripper_motor_joint", "r_gripper_motor_joint"]
CAMERAS = {"ego_view": "head_rgb", "left_wrist": "left_wrist_rgb",
           "right_wrist": "right_wrist_rgb"}
VECTOR_NAMES = {
    "observation.state": JOINTS,
    "observation.projected_gravity": ["gravity_x", "gravity_y", "gravity_z"],
    "action.motion_token": [f"token_{i}" for i in range(64)],
    "action.wbc": JOINTS,
}


def modality():
    return json.loads((ROOT / "modality.json").read_text())
