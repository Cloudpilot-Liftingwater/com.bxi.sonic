# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Native physical-1DoF ELF3 modality configuration for GR00T N1.7.

This is a new training ABI. It never aliases or rewrites the released 68D
ELF3 checkpoint ABI registered by ``examples/ELF3/elf3_sonic_config.py``.
"""

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)


VIDEO_KEYS = ["ego_view", "left_wrist", "right_wrist"]
STATE_KEYS = [
    "waist",
    "left_leg",
    "right_leg",
    "left_arm",
    "right_arm",
    "left_gripper",
    "right_gripper",
    "projected_gravity",
]
ACTION_KEYS = ["motion_token", "left_gripper", "right_gripper"]
LANGUAGE_KEYS = ["annotation.human.task_description"]
ACTION_HORIZON = 40


elf3_native_1d_config = {
    "video": ModalityConfig(delta_indices=[0], modality_keys=VIDEO_KEYS),
    "state": ModalityConfig(delta_indices=[0], modality_keys=STATE_KEYS),
    "action": ModalityConfig(
        delta_indices=list(range(ACTION_HORIZON)),
        modality_keys=ACTION_KEYS,
        action_configs=[
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            )
            for _ in ACTION_KEYS
        ],
    ),
    "language": ModalityConfig(delta_indices=[0], modality_keys=LANGUAGE_KEYS),
}

register_modality_config(
    elf3_native_1d_config,
    embodiment_tag=EmbodimentTag.ELF3_NATIVE_1D,
)
