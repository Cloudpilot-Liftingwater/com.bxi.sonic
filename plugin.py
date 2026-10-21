from bxi_example_py_elf3.framework.mod_api import (
    ModDefinition,
    ModLoadContext,
    ResourceKey,
    ResourceLoadContext,
    StateBuildContext,
)

from .policy import SonicTeleopPolicy
from .state import SonicTeleopState


SONIC_POLICY = ResourceKey[SonicTeleopPolicy]("com.bxi.sonic/policy")


def _load_policy(context: ResourceLoadContext) -> SonicTeleopPolicy:
    return SonicTeleopPolicy(
        str(context.asset("assets/sonic_action_token.onnx")),
        str(context.asset("assets/stream_reference.npz")),
    )


def _build_state(
    state: StateBuildContext,
    policy,
) -> SonicTeleopState:
    backend = state.string_param('gripper_backend', 'dm')
    if backend not in ('dm', 'bxi_demo'):
        raise ValueError('gripper_backend must be dm or bxi_demo')
    state_type = SonicTeleopState
    if backend == 'bxi_demo':
        from .bxi_demo import BxiDemoState
        state_type = BxiDemoState
    return state_type(
        state.name,
        state.state_id,
        policy,
        require_live_reference=state.bool_param(
            "require_live_reference",
            False,
        ),
        yaw_bias_rad=state.float_param("yaw_bias_rad", 1.57079632679),
        live_reference_timeout_s=state.float_param("live_reference_timeout_s", 0.5),
        idle_frame_start=state.int_param("idle_frame_start", 3509),
        source_blend_seconds=state.float_param("source_blend_seconds", 0.4),
        head_control_enabled=state.bool_param("head_control_enabled", True),
        head_pitch_limit_rad=state.float_param("head_pitch_limit_rad", 0.5),
        head_yaw_limit_rad=state.float_param("head_yaw_limit_rad", 1.0),
        head_pitch_speed_rad_s=state.float_param(
            "head_pitch_speed_rad_s", 1.5
        ),
        head_yaw_speed_rad_s=state.float_param("head_yaw_speed_rad_s", 2.0),
        head_deadband_rad=state.float_param("head_deadband_rad", 0.015),
        hardware_gripper=state.bool_param("hardware_gripper", False),
        gripper_enable_interval_s=state.float_param(
            "gripper_enable_interval_s",
            1.0,
        ),
        gripper_detection_interval_s=state.float_param(
            "gripper_detection_interval_s",
            0.25,
        ),
        gripper_detection_timeout_s=state.float_param(
            "gripper_detection_timeout_s",
            3.0,
        ),
        gripper_left_bus=state.int_param("gripper_left_bus", 5),
        gripper_right_bus=state.int_param("gripper_right_bus", 6),
        gripper_left_protocol=state.string_param(
            "gripper_left_protocol", "dm"
        ),
        gripper_right_protocol=state.string_param(
            "gripper_right_protocol", "dm"
        ),
        gripper_dm_can_id=state.int_param("gripper_dm_can_id", 4),
        gripper_dm_master_id=state.int_param("gripper_dm_master_id", 0x14),
        gripper_dm_frame_flags=state.int_param("gripper_dm_frame_flags", 0),
        gripper_discovery_only=state.bool_param("gripper_discovery_only", True),
        gripper_enable_on_discovery=state.bool_param(
            "gripper_enable_on_discovery", True
        ),
        gripper_dm_scan=state.bool_param("gripper_dm_scan", False),
        gripper_dm_scan_max_id=state.int_param("gripper_dm_scan_max_id", 15),
        gripper_dm_auto_master_id=state.bool_param("gripper_dm_auto_master_id", False),
        gripper_left_dm_can_id=state.int_param("gripper_left_dm_can_id", -1),
        gripper_right_dm_can_id=state.int_param("gripper_right_dm_can_id", -1),
        gripper_left_dm_master_id=state.int_param("gripper_left_dm_master_id", -1),
        gripper_right_dm_master_id=state.int_param("gripper_right_dm_master_id", -1),
        gripper_left_dm_frame_flags=state.int_param("gripper_left_dm_frame_flags", -1),
        gripper_right_dm_frame_flags=state.int_param("gripper_right_dm_frame_flags", -1),
        gripper_motion_enabled=state.bool_param("gripper_motion_enabled", False),
        gripper_auto_calibrate=state.bool_param("gripper_auto_calibrate", False),
        gripper_left_open_position_rad=state.float_param(
            "gripper_left_open_position_rad", 0.0
        ),
        gripper_left_closed_position_rad=state.float_param(
            "gripper_left_closed_position_rad", 0.0
        ),
        gripper_right_open_position_rad=state.float_param(
            "gripper_right_open_position_rad", 0.0
        ),
        gripper_right_closed_position_rad=state.float_param(
            "gripper_right_closed_position_rad", 0.0
        ),
        gripper_target_speed_rad_s=state.float_param(
            "gripper_target_speed_rad_s", 0.5
        ),
        gripper_trigger_timeout_s=state.float_param(
            "gripper_trigger_timeout_s", 0.25
        ),
        gripper_pause_with_live_reference=state.bool_param(
            "gripper_pause_with_live_reference", True
        ),
        gripper_motion_torque_limit_nm=state.float_param(
            "gripper_motion_torque_limit_nm", 4.0
        ),
        gripper_motion_command_torque_nm=state.float_param("gripper_motion_command_torque_nm", 2.5),
        gripper_hold_torque_nm=state.float_param("gripper_hold_torque_nm", 2.0),
        gripper_motion_contact_torque_nm=state.float_param("gripper_motion_contact_torque_nm", 1.8),
        gripper_motion_soft_torque_nm=state.float_param("gripper_motion_soft_torque_nm", 3.0),
        gripper_motion_acceleration_rad_s2=state.float_param("gripper_motion_acceleration_rad_s2", 1.0),
        gripper_motion_braking_acceleration_rad_s2=state.float_param("gripper_motion_braking_acceleration_rad_s2", 1.0),
        gripper_motion_velocity_feedforward=state.bool_param("gripper_motion_velocity_feedforward", False),
        gripper_motion_maximum_feedback_velocity_rad_s=state.float_param("gripper_motion_maximum_feedback_velocity_rad_s", 2.0),
        gripper_motion_stall_timeout_s=state.float_param(
            "gripper_motion_stall_timeout_s", 1.5
        ),
        gripper_feedback_jump_limit_rad=state.float_param(
            "gripper_feedback_jump_limit_rad", 1.0
        ),
        gripper_dm_ranges_verified=state.bool_param(
            "gripper_dm_ranges_verified", False
        ),
        gripper_dm_mit_mode_verified=state.bool_param(
            "gripper_dm_mit_mode_verified", False
        ),
        gripper_dm_position_max_rad=state.float_param(
            "gripper_dm_position_max_rad", 12.5
        ),
        gripper_dm_velocity_max_rad_s=state.float_param(
            "gripper_dm_velocity_max_rad_s", 10.0
        ),
        gripper_dm_torque_max_nm=state.float_param(
            "gripper_dm_torque_max_nm", 12.0
        ),
        gripper_kp=state.float_param("gripper_kp", 20.0),
        gripper_kd=state.float_param("gripper_kd", 1.0),
        gripper_calibration_speed_rad_s=state.float_param(
            "gripper_calibration_speed_rad_s",
            0.2,
        ),
        gripper_calibration_kp=state.float_param("gripper_calibration_kp", 5.0),
        gripper_calibration_return_speed_rad_s=state.float_param("gripper_calibration_return_speed_rad_s", .2),
        gripper_calibration_return_acceleration_rad_s2=state.float_param("gripper_calibration_return_acceleration_rad_s2", 1.0),
        gripper_calibration_kd=state.float_param("gripper_calibration_kd", 0.5),
        gripper_calibration_command_torque_limit_nm=state.float_param(
            "gripper_calibration_command_torque_limit_nm", 2.5),
        gripper_feedforward_max_nm=state.float_param("gripper_feedforward_max_nm", 1.5),
        gripper_feedforward_slew_nm_s=state.float_param("gripper_feedforward_slew_nm_s", 1.5),
        gripper_calibration_stall_timeout_s=state.float_param(
            "gripper_calibration_stall_timeout_s", 3.0),
        gripper_minimum_seek_travel_rad=state.float_param(
            "gripper_minimum_seek_travel_rad", 0.1),
        gripper_initial_release_enabled=state.bool_param("gripper_initial_release_enabled", True),
        gripper_initial_release_rad=state.float_param("gripper_initial_release_rad", 0.2),
        gripper_close_on_normal_exit=state.bool_param("gripper_close_on_normal_exit", True),
        gripper_exit_close_speed_rad_s=state.float_param("gripper_exit_close_speed_rad_s", 0.2),
        gripper_exit_close_acceleration_rad_s2=state.float_param("gripper_exit_close_acceleration_rad_s2", 1.0),
        gripper_exit_close_timeout_s=state.float_param("gripper_exit_close_timeout_s", 45.0),
        gripper_left_open_direction=state.int_param("gripper_left_open_direction", 1),
        gripper_right_open_direction=state.int_param("gripper_right_open_direction", 1),
        gripper_contact_torque=state.float_param("gripper_contact_torque", 2.0),
        gripper_abort_torque=state.float_param("gripper_abort_torque", 8.0),
        gripper_contact_confirm_s=state.float_param(
            "gripper_contact_confirm_s",
            0.25,
        ),
        gripper_stopped_velocity_rad_s=state.float_param(
            "gripper_stopped_velocity_rad_s",
            0.1,
        ),
        gripper_tracking_error_rad=state.float_param(
            "gripper_tracking_error_rad",
            0.08,
        ),
        gripper_limit_margin_rad=state.float_param(
            "gripper_limit_margin_rad",
            0.15,
        ),
        gripper_minimum_span_rad=state.float_param(
            "gripper_minimum_span_rad",
            1.0,
        ),
        gripper_maximum_search_travel_rad=state.float_param(
            "gripper_maximum_search_travel_rad",
            7.0,
        ),
        gripper_response_timeout_s=state.float_param(
            "gripper_response_timeout_s",
            1.0,
        ),
        gripper_feedback_timeout_s=state.float_param(
            "gripper_feedback_timeout_s",
            0.3,
        ),
        gripper_phase_timeout_s=state.float_param(
            "gripper_phase_timeout_s",
            45.0,
        ),
        gripper_maximum_mos_temperature_c=state.int_param(
            "gripper_maximum_mos_temperature_c",
            80,
        ),
        gripper_maximum_motor_temperature_c=state.int_param(
            "gripper_maximum_motor_temperature_c",
            80,
        ),
        gripper_maximum_feedback_velocity_rad_s=state.float_param(
            "gripper_maximum_feedback_velocity_rad_s",
            2.0,
        ),
    )


def create_mod(context: ModLoadContext) -> ModDefinition:
    context.register_resource(SONIC_POLICY, _load_policy, policy="startup")
    policy = context.resource(SONIC_POLICY)
    return ModDefinition(
        state_factories={
            "sonic_teleop": lambda state: _build_state(state, policy),
        }
    )


__all__ = ["SONIC_POLICY", "create_mod"]
