#!/usr/bin/env python3
"""Run installed official GR00T loader/statistics/sampler on an exported dataset; no model/robot."""
import argparse
import json
from pathlib import Path

import numpy as np

from contract import ABI, FPS, HORIZON, JOINTS, SCHEMA, VECTOR_NAMES
from elf3_native_1d_config import elf3_native_1d_config as CONFIG


def validate(path):
    import gr00t
    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
    from gr00t.data.dataset.sharded_single_step_dataset import (
        ShardedSingleStepDataset, extract_step_data,
    )
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.data.stats import calculate_dataset_statistics
    from gr00t.data.state_action.state_action_processor import StateActionProcessor
    from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config

    path = Path(path).resolve(strict=True)
    info = json.loads((path / "meta/info.json").read_text())
    contract = json.loads((path / "meta/elf3_contract.json").read_text())
    if (contract.get("gripper_coordinate") != "absolute_motor_radians"
            or contract.get("gripper_position_unit") != "rad"
            or contract.get("joint_names") != JOINTS):
        raise ValueError("Expected real motor radians; simulated opening coordinates are not this dataset")
    if info["fps"] != FPS:
        raise ValueError("This ELF3 project uses 50 Hz; GR00T itself is not fixed to 50 Hz")
    if info["features"]["observation.state"]["shape"] != [31] or info["features"]["action.wbc"]["shape"] != [31]:
        raise ValueError("Expected existing body29 + gripper2 layout")
    from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import EMBODIMENT_TAG_TO_PROJECTOR_INDEX
    if EMBODIMENT_TAG_TO_PROJECTOR_INDEX.get(EmbodimentTag.ELF3_NATIVE_1D.value) != ABI["projector_slot"]:
        raise ValueError("Use the existing ELF3-adapted GR00T repository with native slot14")
    statistics = calculate_dataset_statistics(
        sorted(path.glob("data/*/*.parquet")), features=list(VECTOR_NAMES))
    stored = json.loads((path / "meta/stats.json").read_text())
    for key, value in statistics.items():
        for stat, arr in value.items():
            np.testing.assert_allclose(stored[key][stat], arr, rtol=1e-5, atol=1e-6)
    loader = LeRobotEpisodeLoader(path, CONFIG, decoder_kwargs={"num_ffmpeg_threads": 1})
    loader.get_dataset_statistics()
    rows = 0
    # Decode every image of every video through the actual official loader.
    for i in range(len(loader)):
        df = loader[i]
        rows += len(df)
        assert len(df) >= HORIZON
        assert "state.head" not in df and "action.head_position" not in df
        assert np.stack(df["state.left_gripper"]).shape[1:] == (1,)
        assert np.stack(df["action.motion_token"]).shape[1:] == (64,)
        assert len(df.filter(regex="^video\\.").columns) == 3
    dataset = ShardedSingleStepDataset(
        path, EmbodimentTag.ELF3_NATIVE_1D, CONFIG, episode_sampling_rate=1.0,
        allow_padding=False, shard_size=2048)
    # The official training sampler must materialize genuine 40-step action chunks.
    step = extract_step_data(loader[0], 0, CONFIG, EmbodimentTag.ELF3_NATIVE_1D,
                             allow_padding=False)
    assert sum(x.shape[-1] for x in step.states.values()) == 34
    assert sum(x.shape[-1] for x in step.actions.values()) == 66
    assert all(x.shape[0] == HORIZON for x in step.actions.values())
    tag = EmbodimentTag.ELF3_NATIVE_1D.value
    processor = StateActionProcessor(
        {tag: CONFIG}, {tag: loader.get_dataset_statistics()}, use_relative_action=False)
    states, actions = processor.apply(step.states, step.actions, tag)
    assert all(np.isfinite(x).all() for x in [*states.values(), *actions.values()])
    restored_states, restored_actions = processor.unapply(states, actions, tag)
    for key in ("left_gripper", "right_gripper"):
        np.testing.assert_allclose(restored_states[key], step.states[key], atol=1e-6)
        np.testing.assert_allclose(restored_actions[key], step.actions[key], atol=1e-6)
    limits = Gr00tN1d7Config()
    assert limits.max_state_dim >= 34 and limits.max_action_dim >= 66
    assert limits.action_horizon >= HORIZON
    return dict(
        status="EXISTING_ELF3_NATIVE_DATA_PATH_PASS",
        embodiment_tag=tag, projector_slot=ABI["projector_slot"], gr00t_source=list(gr00t.__path__),
        episodes=len(loader), decoded_video_frames=rows * 3, rows=rows,
        state_shape=[1, 34], action_shape=[HORIZON, 66], views=3,
        test_scope="official loader + CPU video decoding + statistics + training sampling + normalization",
        sampler_steps=int(sum(dataset.shard_lengths)), motor_radian_normalization_roundtrip=True,
        pretrained_processor_run=False,
        finetuning_run=False, hardware_test=False,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("dataset", type=Path)
    a = p.parse_args()
    print(json.dumps(validate(a.dataset), ensure_ascii=False))
