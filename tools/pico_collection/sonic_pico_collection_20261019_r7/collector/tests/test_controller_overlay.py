from __future__ import annotations

import ast
from pathlib import Path
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[1]
SONIC = ROOT / "controller_overlay/src/bxi_example_py_elf3/mods/com.bxi.sonic"


class ControllerOverlayTests(unittest.TestCase):
    def test_policy_uses_one_action_token_backend_run(self):
        path = SONIC / "policy.py"
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        inference_step = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "inference_step"
        )
        backend_runs = [
            node
            for node in ast.walk(inference_step)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "run"
            and isinstance(node.func.value, ast.Attribute)
            and node.func.value.attr == "_backend"
        ]
        self.assertEqual(len(backend_runs), 1)
        source = path.read_text(encoding="utf-8")
        self.assertIn("spec = ModelSpec.onnx(", source)
        self.assertIn('output_names=("action", "motion_token")', source)
        self.assertNotIn("spec = ModelSpec.portable_onnx(", source)

    def test_state_publishes_atomic_token_and_final_gripper_targets(self):
        source = (SONIC / "state.py").read_text(encoding="utf-8")
        self.assertIn('"motion_token": policy.motion_token.tolist()', source)
        self.assertIn('"target_position": np.asarray(', source)
        self.assertIn('GRIPPER_COMMAND_JOINT_NAMES = ("left_gripper", "right_gripper")', source)
        self.assertIn("self._publish_gripper_telemetry(ctx, feedback)", source)
        self.assertIn("SONIC_DISABLE_HARDWARE_GRIPPER", source)

    def test_pico_manager_explains_missing_body_stream_and_button_order(self):
        source = (
            SONIC
            / "pico/gear_sonic/scripts/pico_manager_thread_server.py"
        ).read_text(encoding="utf-8")
        self.assertIn("Calibration complete", source)
        self.assertIn("A+X ignored", source)
        self.assertIn("press Send in the headset app", source)
        self.assertIn("SONIC_PICO_MOCK", source)
        self.assertIn("SIM-ONLY MOCK body stream enabled", source)
        self.assertIn("PICO Y+A: episode start requested", source)
        self.assertIn("PICO Y+B: episode finish requested", source)
        self.assertIn('"start_data_collection": np.array(', source)
        self.assertIn('"finish_data_collection": np.array(', source)
        pose_packet = source.split("numpy_data = {", 1)[1].split(
            "packed_message = pack_pose_message", 1
        )[0]
        self.assertNotIn('"toggle_data_collection"', pose_packet)
        self.assertNotIn('"toggle_data_abort"', pose_packet)
        pose_mode = source.split(
            "elif current_mode == StreamMode.POSE:", 1
        )[1].split(
            "elif current_mode == StreamMode.PLANNER_FROZEN_UPPER_BODY:", 1
        )[0]
        self.assertNotIn("combo.by_rising", pose_mode)

    def test_manifest_exposes_collection_topics(self):
        manifest = yaml.safe_load((SONIC / "mod.yaml").read_text(encoding="utf-8"))
        params = manifest["states"]["sonic_teleop"]["params"]
        self.assertEqual(params["inference_result_topic"], "/sonic/inference_result")
        self.assertEqual(
            params["gripper_state_topic"], "/hardware/gripper_joint_states"
        )
        self.assertEqual(
            params["gripper_command_topic"], "/sonic/gripper_joint_commands"
        )


if __name__ == "__main__":
    unittest.main()
