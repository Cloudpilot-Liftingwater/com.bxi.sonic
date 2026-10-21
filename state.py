from __future__ import annotations

import math
import json
import struct
import time
from dataclasses import replace
from collections import deque
from threading import Lock
from typing import TYPE_CHECKING, Optional, Protocol

import numpy as np
import communication.msg as bxi_msg
from rclpy.qos import QoSProfile
from std_msgs.msg import Float32

from bxi_example_py_elf3.framework.inference import InferenceFrame, PolicyOutput
from bxi_example_py_elf3.framework.mod_api import (
    JointCommandComposer,
    JointCommandLayer,
    JointLayout,
    JointTargetBuffer,
    ResourceHandle,
    RobotControlState,
    StateBehavior,
)
from bxi_example_py_elf3.framework.mod_api.transition import (
    EntryFrameProvider,
    MotorFrame,
    RunningFrameProvider,
)
from bxi_example_py_elf3.policies.joints import ELF3_POLICY_JOINTS

from .gripper import (
    DMMotor,
    CAN_CLASSIC_FLAGS,
    CalibrationPhase,
    CalibrationSettings,
    GripperCalibrator,
    GripperMotorProfile,
    GripperProtocol,
    SUPPORTED_CAN_FRAME_FLAGS,
    JointControl,
    MotorFeedback,
    select_protocol_candidates,
    GripMotionController,
    GripMotionSettings,
)

if TYPE_CHECKING:
    from bxi_example_py_elf3.framework.mod_api import LoggerLike, RobotControlContext


HEAD_JOINT_NAMES = ("head_y_joint", "head_z_joint")
SONIC_HEAD_JOINTS = JointLayout(HEAD_JOINT_NAMES, label="SONIC PICO head command")
GRIPPER_RELEASE = "20261014-reset-dm-only-no-bxi-integration"
SONIC_OUTPUT_JOINTS = JointLayout(
    (*ELF3_POLICY_JOINTS.names, *SONIC_HEAD_JOINTS.names),
    label="SONIC state output",
)


class SonicPolicy(Protocol):
    output: PolicyOutput
    last_status: str
    head_joint_target: np.ndarray

    def bind_logger(self, logger: LoggerLike) -> None:
        ...

    def reset(self, frame: InferenceFrame | None = None) -> None:
        ...

    def step(
        self,
        frame: InferenceFrame,
        dt: float,
        *,
        advance: bool = True,
    ) -> PolicyOutput:
        ...

    def configure_runtime(
        self,
        *,
        yaw_bias_rad: float,
        live_ref_timeout_s: float,
        idle_frame_start: int,
        source_blend_duration_s: float,
    ) -> None:
        ...

    def has_fresh_live_reference(self, timeout_s: float | None = None) -> bool:
        ...

    def reset_yaw_alignment(self) -> None:
        ...


class SonicTeleopState(
    RobotControlState,
    EntryFrameProvider,
    RunningFrameProvider,
):
    """Named-joint SONIC policy state with PICO head and optional gripper control."""

    HEAD_KP = 16.747
    HEAD_KD = 1.066

    def __init__(
        self,
        name: str,
        state_id: int,
        policy: ResourceHandle[SonicPolicy],
        *,
        require_live_reference: bool = False,
        yaw_bias_rad: float = math.pi / 2.0,
        live_reference_timeout_s: float = 0.5,
        idle_frame_start: int = 3509,
        source_blend_seconds: float = 0.4,
        head_control_enabled: bool = True,
        head_pitch_limit_rad: float = 0.5,
        head_yaw_limit_rad: float = 1.0,
        head_pitch_speed_rad_s: float = 1.5,
        head_yaw_speed_rad_s: float = 2.0,
        head_deadband_rad: float = 0.015,
        hardware_gripper: bool = False,
        gripper_enable_interval_s: float = 1.0,
        gripper_detection_interval_s: float = 0.25,
        gripper_detection_timeout_s: float = 3.0,
        gripper_left_bus: int = 5,
        gripper_right_bus: int = 6,
        gripper_left_protocol: str = "dm",
        gripper_right_protocol: str = "dm",
        gripper_dm_can_id: int = 4,
        gripper_dm_master_id: int = 0x14,
        gripper_dm_frame_flags: int = CAN_CLASSIC_FLAGS,
        gripper_discovery_only: bool = True,
        gripper_enable_on_discovery: bool = True,
        gripper_dm_scan: bool = False,
        gripper_dm_scan_max_id: int = 15,
        gripper_dm_auto_master_id: bool = False,
        gripper_left_dm_can_id: int = -1,
        gripper_right_dm_can_id: int = -1,
        gripper_left_dm_master_id: int = -1,
        gripper_right_dm_master_id: int = -1,
        gripper_left_dm_frame_flags: int = -1,
        gripper_right_dm_frame_flags: int = -1,
        gripper_motion_enabled: bool = False,
        gripper_auto_calibrate: bool = False,
        gripper_left_open_position_rad: float = 0.0,
        gripper_left_closed_position_rad: float = 0.0,
        gripper_right_open_position_rad: float = 0.0,
        gripper_right_closed_position_rad: float = 0.0,
        gripper_target_speed_rad_s: float = 0.5,
        gripper_trigger_timeout_s: float = 0.25,
        gripper_pause_with_live_reference: bool = True,
        gripper_motion_torque_limit_nm: float = 4.0,
        gripper_motion_command_torque_nm: float = 2.5,
        gripper_hold_torque_nm: float = 2.0,
        gripper_motion_contact_torque_nm: float = 1.8,
        gripper_motion_soft_torque_nm: float = 3.0,
        gripper_motion_acceleration_rad_s2: float = 1.0,
        gripper_motion_braking_acceleration_rad_s2: float = 1.0,
        gripper_motion_velocity_feedforward: bool = False,
        gripper_motion_maximum_feedback_velocity_rad_s: float = 2.0,
        gripper_motion_stall_timeout_s: float = 1.5,
        gripper_feedback_jump_limit_rad: float = 1.0,
        gripper_dm_ranges_verified: bool = False,
        gripper_dm_mit_mode_verified: bool = False,
        gripper_dm_position_max_rad: float = 12.5,
        gripper_dm_velocity_max_rad_s: float = 10.0,
        gripper_dm_torque_max_nm: float = 12.0,
        gripper_kp: float = 20.0,
        gripper_kd: float = 1.0,
        gripper_calibration_speed_rad_s: float = 0.2,
        gripper_calibration_return_speed_rad_s: float = 0.2,
        gripper_calibration_return_acceleration_rad_s2: float = 1.0,
        gripper_calibration_kp: float = 5.0,
        gripper_calibration_kd: float = 0.5,
        gripper_calibration_command_torque_limit_nm: float = 2.5,
        gripper_feedforward_max_nm: float = 1.5,
        gripper_feedforward_slew_nm_s: float = 1.5,
        gripper_calibration_stall_timeout_s: float = 3.0,
        gripper_minimum_seek_travel_rad: float = 0.1,
        gripper_initial_release_enabled: bool = True,
        gripper_initial_release_rad: float = 0.2,
        gripper_close_on_normal_exit: bool = True,
        gripper_exit_close_speed_rad_s: float = 0.2,
        gripper_exit_close_acceleration_rad_s2: float = 1.0,
        gripper_exit_close_timeout_s: float = 45.0,
        gripper_left_open_direction: int = 1,
        gripper_right_open_direction: int = 1,
        gripper_contact_torque: float = 2.0,
        gripper_abort_torque: float = 8.0,
        gripper_contact_confirm_s: float = 0.25,
        gripper_stopped_velocity_rad_s: float = 0.1,
        gripper_tracking_error_rad: float = 0.08,
        gripper_limit_margin_rad: float = 0.15,
        gripper_minimum_span_rad: float = 1.0,
        gripper_maximum_search_travel_rad: float = 7.0,
        gripper_response_timeout_s: float = 1.0,
        gripper_feedback_timeout_s: float = 0.3,
        gripper_phase_timeout_s: float = 45.0,
        gripper_maximum_mos_temperature_c: int = 80,
        gripper_maximum_motor_temperature_c: int = 80,
        gripper_maximum_feedback_velocity_rad_s: float = 2.0,
    ) -> None:
        super().__init__(name, state_id, resources=(policy,))
        if gripper_enable_interval_s <= 0.0:
            raise ValueError("gripper_enable_interval_s must be positive")
        if gripper_detection_interval_s <= 0.0:
            raise ValueError("gripper_detection_interval_s must be positive")
        if gripper_detection_timeout_s <= gripper_detection_interval_s:
            raise ValueError(
                "gripper_detection_timeout_s must exceed the probe interval"
            )
        self._policy = policy
        self.require_live_reference = bool(require_live_reference)
        self.yaw_bias_rad = float(yaw_bias_rad)
        self.live_reference_timeout_s = float(live_reference_timeout_s)
        self.idle_frame_start = int(idle_frame_start)
        self.source_blend_seconds = float(source_blend_seconds)
        self.head_control_enabled = bool(head_control_enabled)
        self.head_pitch_limit_rad = float(head_pitch_limit_rad)
        self.head_yaw_limit_rad = float(head_yaw_limit_rad)
        self.head_pitch_speed_rad_s = float(head_pitch_speed_rad_s)
        self.head_yaw_speed_rad_s = float(head_yaw_speed_rad_s)
        self.head_deadband_rad = float(head_deadband_rad)
        self.hardware_gripper = bool(hardware_gripper)
        self.gripper_enable_interval_s = float(gripper_enable_interval_s)
        self.gripper_detection_interval_s = float(gripper_detection_interval_s)
        self.gripper_detection_timeout_s = float(gripper_detection_timeout_s)
        self._left_bus = int(gripper_left_bus)
        self._right_bus = int(gripper_right_bus)
        self._gripper_discovery_only = bool(gripper_discovery_only)
        self._gripper_enable_on_discovery = bool(gripper_enable_on_discovery)
        self._gripper_dm_scan = bool(gripper_dm_scan)
        self._gripper_dm_auto_master_id = bool(gripper_dm_auto_master_id)
        self._gripper_motion_enabled = bool(gripper_motion_enabled)
        self._gripper_auto_calibrate = bool(gripper_auto_calibrate)
        self._gripper_soft_limits = {
            "left": (
                float(gripper_left_open_position_rad),
                float(gripper_left_closed_position_rad),
            ),
            "right": (
                float(gripper_right_open_position_rad),
                float(gripper_right_closed_position_rad),
            ),
        }
        self._gripper_target_speed_rad_s = float(gripper_target_speed_rad_s)
        self._gripper_trigger_timeout_s = float(gripper_trigger_timeout_s)
        self._gripper_pause_with_live_reference = bool(
            gripper_pause_with_live_reference
        )
        self._gripper_motion_torque_limit_nm = float(gripper_motion_torque_limit_nm)
        self._gripper_grip_config = dict(
            command_nm=float(gripper_motion_command_torque_nm),
            hold_nm=float(gripper_hold_torque_nm),
            contact_nm=float(gripper_motion_contact_torque_nm),
            soft_nm=float(gripper_motion_soft_torque_nm),
            hard_nm=float(gripper_motion_torque_limit_nm),
            acceleration=float(gripper_motion_acceleration_rad_s2),
            braking_acceleration=float(gripper_motion_braking_acceleration_rad_s2),
            velocity_feedforward=gripper_motion_velocity_feedforward)
        self._gripper_motion_velocity_limit = float(gripper_motion_maximum_feedback_velocity_rad_s)
        self._gripper_exit_acceleration = float(gripper_exit_close_acceleration_rad_s2)
        if not all(math.isfinite(v) and v > 0 for v in
                   (self._gripper_motion_velocity_limit, self._gripper_exit_acceleration)):
            raise ValueError("motion feedback speed and exit acceleration must be positive and finite")
        if not self._gripper_target_speed_rad_s < self._gripper_motion_velocity_limit < gripper_dm_velocity_max_rad_s:
            raise ValueError("require motion speed < feedback speed guard < DM codec VMAX")
        self._gripper_motion_stall_timeout_s = float(gripper_motion_stall_timeout_s)
        self._gripper_feedback_jump_limit_rad = float(gripper_feedback_jump_limit_rad)
        self._gripper_dm_ranges_verified = bool(gripper_dm_ranges_verified)
        self._gripper_dm_mit_mode_verified = bool(gripper_dm_mit_mode_verified)
        scan_max_id = int(gripper_dm_scan_max_id)
        if not 0 <= scan_max_id <= 15:
            raise ValueError("gripper_dm_scan_max_id must be in 0..15")
        # Finding an unknown motor must not implicitly enable it with assumed
        # MIT ranges/mode.  Confirm those parameters before explicit control.
        if self._gripper_dm_scan and not self._gripper_discovery_only:
            raise ValueError("DM ID scan requires gripper_discovery_only=true")
        if self._gripper_motion_enabled and self._gripper_discovery_only:
            raise ValueError(
                "gripper_motion_enabled requires gripper_discovery_only=false"
            )
        if self._gripper_auto_calibrate and self._gripper_discovery_only:
            raise ValueError(
                "gripper_auto_calibrate requires gripper_discovery_only=false"
            )
        if self._gripper_auto_calibrate and not self._gripper_motion_enabled:
            raise ValueError(
                "gripper_auto_calibrate requires gripper_motion_enabled=true"
            )
        if not self._gripper_discovery_only and not self._gripper_motion_enabled:
            raise ValueError(
                "active gripper control requires gripper_motion_enabled=true"
            )
        if self._gripper_motion_enabled and self._gripper_dm_scan:
            raise ValueError(
                "gripper_motion_enabled requires explicit motor identities; "
                "disable gripper_dm_scan"
            )
        if self._gripper_motion_enabled and self._gripper_dm_auto_master_id:
            raise ValueError(
                "gripper_motion_enabled requires fixed master IDs; "
                "disable gripper_dm_auto_master_id"
            )
        self._gripper_auto_buses = {
            bus for bus, mode in ((self._left_bus, gripper_left_protocol),
                                  (self._right_bus, gripper_right_protocol))
            if str(mode).strip().lower() == "auto"
        }
        if (
            self._gripper_motion_enabled
            and (
                str(gripper_left_protocol).strip().lower() in ("dm", "auto")
                or str(gripper_right_protocol).strip().lower() in ("dm", "auto")
            )
            and not self._gripper_dm_ranges_verified
        ):
            raise ValueError(
                "DM gripper motion requires verified PMAX/VMAX/TMAX ranges"
            )
        if (
            self._gripper_motion_enabled
            and (
                str(gripper_left_protocol).strip().lower() in ("dm", "auto")
                or str(gripper_right_protocol).strip().lower() in ("dm", "auto")
            )
            and not self._gripper_dm_mit_mode_verified
        ):
            raise ValueError(
                "DM gripper motion requires verified CTRL_MODE=MIT"
            )
        for side, protocol, can_id, master_id in (
            ("left", gripper_left_protocol, gripper_left_dm_can_id, gripper_left_dm_master_id),
            ("right", gripper_right_protocol, gripper_right_dm_can_id, gripper_right_dm_master_id),
        ):
            if (
                self._gripper_motion_enabled
                and str(protocol).strip().lower() in ("dm", "auto")
                and (int(can_id) < 0 or int(master_id) < 0)
            ):
                raise ValueError(
                    f"{side} DM gripper motion requires explicit CAN ID and master ID"
                )
        if self._gripper_motion_enabled and not self._gripper_enable_on_discovery:
            raise ValueError(
                "gripper_motion_enabled requires gripper_enable_on_discovery=true"
            )
        if self._left_bus == self._right_bus:
            raise ValueError("left and right gripper buses must differ")

        def port_candidates(
            label: str,
            bus: int,
            mode: str,
            dm_can_id: int,
            dm_master_id: int,
            dm_frame_flags: int,
        ) -> tuple[GripperMotorProfile, ...]:
            selected = select_protocol_candidates(
                mode,
                (
                    GripperMotorProfile(
                        label=label,
                        protocol="dm",
                        bus=bus,
                        can_id=int(gripper_dm_can_id) if dm_can_id == -1 else int(dm_can_id),
                        master_id=int(gripper_dm_master_id) if dm_master_id == -1 else int(dm_master_id),
                        frame_flags=int(gripper_dm_frame_flags) if dm_frame_flags == -1 else int(dm_frame_flags),
                        position_max_rad=float(gripper_dm_position_max_rad),
                        velocity_max_rad_s=float(gripper_dm_velocity_max_rad_s),
                        torque_max_nm=float(gripper_dm_torque_max_nm),
                    ),
                ),
            )
            expanded = []
            for profile in selected:
                expanded.append(profile)  # configured ID first
                if self._gripper_dm_scan and profile.protocol is GripperProtocol.DM:
                    expanded.extend(
                        replace(profile, can_id=can_id)
                        for can_id in range(scan_max_id + 1)
                        if can_id != profile.can_id
                    )
            return tuple(expanded)

        self._gripper_candidates = {
            self._left_bus: port_candidates(
                "left", self._left_bus, gripper_left_protocol,
                gripper_left_dm_can_id, gripper_left_dm_master_id,
                gripper_left_dm_frame_flags,
            ),
            self._right_bus: port_candidates(
                "right", self._right_bus, gripper_right_protocol,
                gripper_right_dm_can_id, gripper_right_dm_master_id,
                gripper_right_dm_frame_flags,
            ),
        }
        required_window = (
            max(map(len, self._gripper_candidates.values())) + 1
        ) * self.gripper_detection_interval_s
        if self.gripper_detection_timeout_s < required_window:
            raise ValueError(
                f"gripper_detection_timeout_s must cover a full scan and reply "
                f"grace: at least {required_window:.2f}s"
            )
        # Populated only by strict read-only replies at the start of each SONIC
        # session.  Once both ports are selected, their profiles stay locked
        # until on_exit() and are used for enable, feedback and MIT control.
        self._gripper_profiles: dict[int, GripperMotorProfile] = {}
        self._gripper_kp = float(gripper_kp)
        self._gripper_kd = float(gripper_kd)
        self._gripper_calibration_kp = float(gripper_calibration_kp)
        self._gripper_calibration_kd = float(gripper_calibration_kd)
        self._gripper_open_directions = {
            self._left_bus: gripper_left_open_direction,
            self._right_bus: gripper_right_open_direction,
        }
        self._gripper_calibration_kwargs = {
            "initial_release_enabled": gripper_initial_release_enabled,
            "initial_release_rad": float(gripper_initial_release_rad),
            "kp": self._gripper_calibration_kp,
            "kd": self._gripper_calibration_kd,
            "command_torque_limit_nm": float(gripper_calibration_command_torque_limit_nm),
            "feedforward_max_nm": float(gripper_feedforward_max_nm),
            "feedforward_slew_nm_s": float(gripper_feedforward_slew_nm_s),
            "stall_timeout_s": float(gripper_calibration_stall_timeout_s),
            "minimum_seek_travel_rad": float(gripper_minimum_seek_travel_rad),
            "speed_rad_s": float(gripper_calibration_speed_rad_s),
            "return_speed_rad_s": float(gripper_calibration_return_speed_rad_s),
            "return_acceleration_rad_s2": float(gripper_calibration_return_acceleration_rad_s2),
            "contact_torque": float(gripper_contact_torque),
            "abort_torque": float(gripper_abort_torque),
            "contact_confirm_s": float(gripper_contact_confirm_s),
            "stopped_velocity_rad_s": float(gripper_stopped_velocity_rad_s),
            "tracking_error_rad": float(gripper_tracking_error_rad),
            "limit_margin_rad": float(gripper_limit_margin_rad),
            "minimum_span_rad": float(gripper_minimum_span_rad),
            "maximum_search_travel_rad": float(
                gripper_maximum_search_travel_rad
            ),
            "response_timeout_s": float(gripper_response_timeout_s),
            "feedback_timeout_s": float(gripper_feedback_timeout_s),
            "phase_timeout_s": float(gripper_phase_timeout_s),
            "maximum_mos_temperature_c": int(
                gripper_maximum_mos_temperature_c
            ),
            "maximum_motor_temperature_c": int(
                gripper_maximum_motor_temperature_c
            ),
            "maximum_feedback_velocity_rad_s": float(
                gripper_maximum_feedback_velocity_rad_s
            ),
        }
        self._gripper_calibrators: dict[int, GripperCalibrator] = {}
        if not isinstance(gripper_close_on_normal_exit, bool):
            raise ValueError("gripper_close_on_normal_exit must be boolean")
        if not all(math.isfinite(v) and v > 0 for v in
                   (gripper_exit_close_speed_rad_s, gripper_exit_close_timeout_s)):
            raise ValueError("gripper exit speed/timeout must be finite and positive")
        if gripper_exit_close_speed_rad_s > gripper_target_speed_rad_s:
            raise ValueError("gripper exit speed must not exceed normal motion speed")
        self._gripper_close_on_normal_exit = gripper_close_on_normal_exit
        self._gripper_exit_speed = float(gripper_exit_close_speed_rad_s)
        self._gripper_exit_timeout = float(gripper_exit_close_timeout_s)
        self._gripper_exit_started_at: Optional[float] = None
        self._gripper_exit_settled_at: Optional[float] = None
        self._gripper_exit_complete = False
        self._validate_config()
        self._last_running_frame: Optional[MotorFrame] = None
        self._policy_logger_bound = False
        self._head_command = JointTargetBuffer(SONIC_HEAD_JOINTS)
        self._command_composer: JointCommandComposer | None = None

        self._gripper_session_active = False
        self._gripper_armed = False
        self._gripper_calibrated = False
        self._gripper_faulted = False
        self._gripper_fault_state_requested = False
        self._gripper_motion_ready = False
        self._gripper_motion_started_at: Optional[float] = None
        self._last_gripper_enable_time: Optional[float] = None
        self._gripper_detection_started_at = 0.0
        self._last_gripper_detection_time: Optional[float] = None
        self._detected_gripper_buses: set[int] = set()
        self._gripper_detection_errors: dict[int, str] = {}
        self._gripper_feedback_errors: dict[int, str] = {}
        self._gripper_detection_cursor = {bus: 0 for bus in self._gripper_candidates}
        self._gripper_sent_probes: set[tuple[int, GripperProtocol, int]] = set()
        self._gripper_rx_counts = {bus: 0 for bus in self._gripper_candidates}
        self._gripper_discovery_finished = False
        self._left_trigger = 0.0
        self._right_trigger = 0.0
        self._left_trigger_received_at: Optional[float] = None
        self._right_trigger_received_at: Optional[float] = None
        self._gripper_trigger_seen: set[str] = set()
        self._gripper_trigger_stale_logged: set[str] = set()
        self._gripper_target_positions: dict[int, float] = {}
        self._gripper_last_feedback_position: dict[int, float] = {}
        self._gripper_motion_feedback_at: dict[int, float] = {}
        self._gripper_motion_controllers: dict[int, GripMotionController] = {}
        self._gripper_recent_motion = deque(maxlen=40)
        self._gripper_recent_motion_at = -math.inf
        self._gripper_feedback_stall_since: dict[int, Optional[float]] = {}
        self._gripper_pose_paused = False
        self._gripper_subscriptions = []
        self._gripper_publisher = None
        self._gripper_available = not self.hardware_gripper
        self._gripper_feedback_lock = Lock()
        self._gripper_input_lock = Lock()
        self._gripper_control_snapshot = None
        self._deferred_diagnostics = None
        self._gripper_feedback: dict[int, MotorFeedback] = {}
        self._gripper_phase_snapshot: dict[int, CalibrationPhase] = {}
        self._gripper_diagnostic_at: dict[int, float] = {}
        self._gripper_register_values: dict[tuple[int, int], float | int] = {}
        self._gripper_register_requested: set[tuple[int, int]] = set()
        self._gripper_register_started_at: Optional[float] = None
        self._gripper_register_last_tx: Optional[float] = None
        self._bad_gripper_feedback_warned: set[int] = set()
        self._collection_telemetry = None
        self._collection_error_at = -math.inf
    def _motion_log(self, level, message, payload=None, key=None):
        worker = self._deferred_diagnostics
        if worker is not None:
            worker.submit(level, message, payload, key)
        else:
            # Offline harness/unbound state only. Real ROS binding starts worker.
            getattr(self.logger, level)(message + (json.dumps(payload, ensure_ascii=False)
                                                  if payload is not None else ''))

    def _clear_gripper_inputs(self):
        with self._gripper_input_lock:
            self._left_trigger_received_at = self._right_trigger_received_at = None
            self._gripper_trigger_seen.clear()


    def _collection_call(self, method, *args):
        # Collection is observational: a serialization/ROS error must never
        # enter the gripper control fault/zero-torque path.
        if method == 'reset':
            self._gripper_control_snapshot = None
        tap = getattr(self, "_collection_telemetry", None)
        if tap is None:
            return
        try:
            getattr(tap, method)(*args)
        except Exception as exc:
            now = time.monotonic()
            if now-getattr(self, "_collection_error_at", -math.inf) >= 5:
                self._collection_error_at = now
                self.logger.warning(f"SONIC数采旁路异常，不能据此判定数采可用：{exc}")

    @property
    def policy(self) -> SonicPolicy:
        return self._policy.get()

    def on_bind(self, ctx: RobotControlContext) -> None:
        from .diagnostics import DeferredDiagnostics
        self._deferred_diagnostics = DeferredDiagnostics(self.logger)
        from .collection_telemetry import CollectionTelemetry
        try:
            self._collection_telemetry = CollectionTelemetry(ctx.ros_node)
        except Exception as exc:
            self.logger.warning(f"SONIC数采接口初始化失败：{exc}")
        if not self.hardware_gripper:
            return
        packet_type = getattr(
            bxi_msg,
            "CANFDPacket",
            getattr(bxi_msg, "CanfdPacket", None),
        )
        if packet_type is None:
            # The gripper is an optional SONIC peripheral.  Some simulation
            # and deployment message packages do not expose the CAN FD packet
            # type, so degrade to body-only teleoperation instead of making
            # the complete SONIC state unavailable.
            self.hardware_gripper = False
            self._gripper_available = True
            self.logger.warning(
                "SONIC夹爪已禁用：缺少communication.msg.CANFDPacket；"
                "全身遥操仍可用"
            )
            return
        qos = QoSProfile(depth=1)
        self._gripper_subscriptions = [
            ctx.ros_node.create_subscription(
                Float32,
                "pico/left_trigger",
                self._left_trigger_callback,
                qos,
            ),
            ctx.ros_node.create_subscription(
                Float32,
                "pico/right_trigger",
                self._right_trigger_callback,
                qos,
            ),
            ctx.ros_node.create_subscription(
                packet_type,
                "canfd_packet/rx",
                self._gripper_feedback_callback,
                QoSProfile(depth=100),
            ),
        ]
        self._gripper_publisher = ctx.ros_node.create_publisher(
            packet_type,
            "canfd_packet/tx",
            QoSProfile(depth=100),
        )
        self._gripper_available = True

    def on_unbind(self, ctx: RobotControlContext) -> None:
        if self._deferred_diagnostics is not None:
            self._deferred_diagnostics.close()
            self._deferred_diagnostics = None
        self._collection_call("close")
        self._collection_telemetry = None
        if self._gripper_armed and self._gripper_publisher is not None:
            self._disable_grippers(repeats=3)
        self._gripper_armed = False
        self._gripper_session_active = False
        for subscription in self._gripper_subscriptions:
            ctx.ros_node.destroy_subscription(subscription)
        self._gripper_subscriptions.clear()
        with self._gripper_feedback_lock:
            self._gripper_feedback.clear()
            self._gripper_feedback_errors.clear()
        if self._gripper_publisher is not None:
            ctx.ros_node.destroy_publisher(self._gripper_publisher)
            self._gripper_publisher = None

    def _calibration_settings_for(
        self,
        profile: GripperMotorProfile,
    ) -> CalibrationSettings:
        settings = CalibrationSettings(
            **self._gripper_calibration_kwargs,
            open_direction=self._gripper_open_directions[profile.bus],
            position_min_rad=-profile.position_max_rad,
            position_max_rad=profile.position_max_rad,
        )
        return settings


    def _create_gripper_calibrators(self, now: float) -> None:
        self._gripper_calibrators = {
            bus: GripperCalibrator(profile.label, self._calibration_settings_for(profile))
            for bus, profile in self._gripper_profiles.items()
        }
        self._gripper_phase_snapshot = {}
        for bus, calibrator in self._gripper_calibrators.items():
            calibrator.reset(now)
            self._gripper_phase_snapshot[bus] = calibrator.phase
            profile = self._gripper_profiles[bus]

    def _soft_limits_for(
        self,
        bus: int,
        profile: GripperMotorProfile,
    ) -> tuple[float, float]:
        side = "left" if bus == self._left_bus else "right"
        if self._gripper_auto_calibrate:
            calibrator = self._gripper_calibrators.get(bus)
            if calibrator is None or not calibrator.ready:
                raise ValueError(f"{side} gripper calibration is not ready")
            assert calibrator.open_position is not None
            assert calibrator.closed_position is not None
            return calibrator.open_position, calibrator.closed_position
        open_position, closed_position = self._gripper_soft_limits[side]
        minimum, maximum = profile.position_range
        if not (minimum <= open_position <= maximum and minimum <= closed_position <= maximum):
            raise ValueError(
                f"{side} gripper soft limits [{open_position}, {closed_position}] "
                f"are outside protocol range [{minimum}, {maximum}]"
            )
        if abs(open_position - closed_position) < 0.1:
            raise ValueError(f"{side} gripper soft limit span is too small")
        return open_position, closed_position

    def _validate_config(self) -> None:
        self._motion_settings().validate()
        if self._left_bus == self._right_bus:
            raise ValueError("left and right gripper buses must differ")
        if set(self._gripper_candidates) != {self._left_bus, self._right_bus}:
            raise ValueError("each gripper bus must have protocol candidates")
        for candidates in self._gripper_candidates.values():
            for profile in candidates:
                self._calibration_settings_for(profile).validate()
                if self._gripper_motion_enabled and max(
                    self._gripper_motion_torque_limit_nm,
                    self._gripper_calibration_kwargs["command_torque_limit_nm"],
                    self._gripper_calibration_kwargs["abort_torque"],
                ) >= profile.torque_max_nm:
                    raise ValueError("gripper torque limits must be below protocol TMAX")
        finite_values = (
            self.yaw_bias_rad,
            self.live_reference_timeout_s,
            self.source_blend_seconds,
            self.head_pitch_limit_rad,
            self.head_yaw_limit_rad,
            self.head_pitch_speed_rad_s,
            self.head_yaw_speed_rad_s,
            self.head_deadband_rad,
            self.gripper_enable_interval_s,
            self.gripper_detection_interval_s,
            self.gripper_detection_timeout_s,
            self._gripper_kp,
            self._gripper_kd,
            self._gripper_calibration_kp,
            self._gripper_calibration_kd,
            self._gripper_target_speed_rad_s,
            self._gripper_trigger_timeout_s,
            self._gripper_motion_torque_limit_nm,
            self._gripper_motion_stall_timeout_s,
            self._gripper_feedback_jump_limit_rad,
        )
        if not all(math.isfinite(value) for value in finite_values):
            raise ValueError("SONIC numeric parameters must be finite")
        if self.live_reference_timeout_s <= 0.0:
            raise ValueError("live_reference_timeout_s must be positive")
        if self.idle_frame_start < 0:
            raise ValueError("idle_frame_start must be non-negative")
        if self.source_blend_seconds < 0.0:
            raise ValueError("source_blend_seconds must be non-negative")
        if min(
            self.head_pitch_limit_rad,
            self.head_yaw_limit_rad,
            self.head_pitch_speed_rad_s,
            self.head_yaw_speed_rad_s,
        ) <= 0.0:
            raise ValueError("SONIC head limits and speeds must be positive")
        if self.head_deadband_rad < 0.0:
            raise ValueError("SONIC head_deadband_rad must be non-negative")
        if not (0 < self._gripper_kp <= 500 and 0 < self._gripper_kd <= 5):
            raise ValueError("gripper MIT position gains must be in their encoding range with kp>0 and kd>0")
        if not all(math.isfinite(v) for pair in self._gripper_soft_limits.values() for v in pair):
            raise ValueError("gripper soft limits must be finite")
        if min(
            self._gripper_target_speed_rad_s,
            self._gripper_trigger_timeout_s,
            self._gripper_motion_torque_limit_nm,
            self._gripper_motion_stall_timeout_s,
            self._gripper_feedback_jump_limit_rad,
        ) <= 0.0:
            raise ValueError("gripper motion safety limits and timeout must be positive")
        if self._gripper_motion_enabled and not self._gripper_auto_calibrate:
            for side, (open_position, closed_position) in self._gripper_soft_limits.items():
                if open_position == closed_position:
                    raise ValueError(
                        f"{side} gripper soft open/closed positions must differ"
                    )
                if abs(open_position - closed_position) < 0.1:
                    raise ValueError(
                        f"{side} gripper soft open/closed span is too small"
                    )

    def is_available(self, ctx: RobotControlContext) -> bool:
        if not self._gripper_available:
            return False
        if self._policy.status != "ready":
            return True
        return not self.require_live_reference or self.policy.has_fresh_live_reference(
            self.live_reference_timeout_s
        )

    def on_prepare(
        self,
        ctx: RobotControlContext,
        from_state: StateBehavior[RobotControlContext],
    ) -> None:
        if not self._policy_logger_bound:
            self.policy.bind_logger(self.logger)
            self._policy_logger_bound = True
        self.policy.configure_runtime(
            yaw_bias_rad=self.yaw_bias_rad,
            live_ref_timeout_s=self.live_reference_timeout_s,
            idle_frame_start=self.idle_frame_start,
            source_blend_duration_s=self.source_blend_seconds,
        )
        self.policy.reset(ctx.inference_frame)
        self._prepare_command_source()
        self._last_running_frame = None
        self._gripper_session_active = False

    def _prepare_command_source(self) -> None:
        self._head_command.position.fill(0.0)
        self._head_command.kp.fill(self.HEAD_KP)
        self._head_command.kd.fill(self.HEAD_KD)
        if not self.head_control_enabled:
            self._command_composer = JointCommandComposer(
                ELF3_POLICY_JOINTS,
                (JointCommandLayer("sonic_policy", self.policy.output.joints),),
            )
            return
        self._command_composer = JointCommandComposer(
            SONIC_OUTPUT_JOINTS,
            (
                JointCommandLayer("sonic_policy", self.policy.output.joints),
                JointCommandLayer("pico_head", self._head_command.view),
            ),
        )

    def get_entry_frame(self, ctx: RobotControlContext) -> MotorFrame:
        return self._compose_motor_frame()

    def _update_head_command(self, desired: object, dt: float) -> None:
        if not self.head_control_enabled:
            return
        target = np.asarray(desired, dtype=np.float32)
        if target.shape != (2,) or not np.isfinite(target).all():
            raise ValueError("SONIC head target must contain two finite joint angles")
        target = np.clip(
            target,
            (-self.head_pitch_limit_rad, -self.head_yaw_limit_rad),
            (self.head_pitch_limit_rad, self.head_yaw_limit_rad),
        )
        target[np.abs(target) < self.head_deadband_rad] = 0.0
        max_step = np.asarray(
            (
                self.head_pitch_speed_rad_s * dt,
                self.head_yaw_speed_rad_s * dt,
            ),
            dtype=np.float32,
        )
        delta = np.clip(
            target - self._head_command.position,
            -max_step,
            max_step,
        )
        self._head_command.position += delta

    def _compose_motor_frame(self) -> MotorFrame:
        if self._command_composer is None:
            raise RuntimeError("SONIC command composer is not prepared")
        return self._command_composer.compose()

    def sample_running_frame(
        self,
        ctx: RobotControlContext,
        dt: float,
        *,
        advance: bool,
    ) -> MotorFrame:
        if not advance:
            return self._last_running_frame or self.get_entry_frame(ctx)
        output = self.policy.step(ctx.inference_frame, dt, advance=True)
        self._update_head_command(self.policy.head_joint_target, dt)
        frame = self._compose_motor_frame()
        self._collection_call("publish", self, output)
        self._last_running_frame = frame
        return frame

    def on_enter(self, ctx: RobotControlContext) -> None:
        self._collection_call("reset")
        mode = "SONIC遥操（夹爪）" if self.hardware_gripper else "SONIC遥操"
        head_status = (
            "头部跟踪已开启"
            if self.head_control_enabled
            else "头部跟踪已关闭"
        )
        self.logger.info(
            f"{mode}已启动；{head_status}；"
            "PICO同时按住A+B+X+Y请求校准，再按A+X切入实时POSE"
        )
        if self.hardware_gripper:
            self._left_trigger = self._right_trigger = 0.0
            now = time.monotonic()
            self._start_gripper_session(now)
            self._publish_gripper_detection(now)
            if self._gripper_motion_enabled and self._gripper_auto_calibrate:
                motion_description = (
                    "检测成功后执行显式开启的受保护低速自动校准，"
                    "校准完成后发送MIT位置帧"
                )
            elif self._gripper_motion_enabled:
                motion_description = (
                    "检测成功后等待双侧健康反馈，使用显式软限位发送MIT位置帧；"
                    "不执行硬限位自动校准"
                )
            elif self._gripper_discovery_only:
                motion_description = "检测成功后保持使能；未校准、未发送MIT位置帧"
            else:
                motion_description = "检测成功后按现有校准器执行；校准完成后接管trigger"
            self.logger.info(
                f"SONIC夹爪探测：buses={list(self._gripper_candidates)}, "
                f"discovery_only={self._gripper_discovery_only}, "
                f"enable_on_discovery={self._gripper_enable_on_discovery}, "
                f"scan={self._gripper_dm_scan}, "
                f"window={self.gripper_detection_timeout_s:.1f}s；"
                "先发送寄存器读取；" + motion_description
            )

    def on_exit(self, ctx: RobotControlContext) -> None:
        self._collection_call("reset")
        self.cancel_prepare_exit(ctx)
        if self._gripper_armed and self._gripper_publisher is not None:
            self._disable_grippers(repeats=3)
        self._gripper_session_active = False
        self._gripper_armed = False
        self._gripper_calibrated = False
        self._gripper_faulted = False
        self._gripper_fault_state_requested = False
        self._gripper_motion_ready = False
        self._gripper_motion_started_at = None
        self._last_gripper_enable_time = None
        self._last_gripper_detection_time = None
        self._left_trigger_received_at = None
        self._right_trigger_received_at = None
        self._gripper_trigger_seen.clear()
        self._gripper_trigger_stale_logged.clear()
        self._gripper_target_positions.clear()
        self._gripper_last_feedback_position.clear()
        self._gripper_motion_feedback_at.clear()
        self._gripper_motion_controllers.clear()
        self._gripper_feedback_stall_since.clear()
        self._gripper_pose_paused = False
        with self._gripper_feedback_lock:
            self._detected_gripper_buses.clear()
            self._gripper_detection_errors.clear()
            self._gripper_profiles.clear()
            self._gripper_feedback.clear()
            self._gripper_feedback_errors.clear()
        self._gripper_calibrators.clear()
        self._gripper_phase_snapshot.clear()

    def prepare_exit(self, ctx: RobotControlContext, to_state: str, *,
                     trigger: str, force: bool = False) -> bool:
        """Runtime exit gate: only an ordinary SONIC -> normal request may wait.

        Safety/recovery/forced exits and shutdown never initiate closing. This
        method is non-blocking; on_update continues body and gripper control.
        """
        if (force or to_state != "com.bxi.basic_actions/normal"
                or not self._gripper_close_on_normal_exit):
            self.cancel_prepare_exit(ctx)
            return True
        ready = self._gripper_motion_ready and self._gripper_calibrated
        if not (self.hardware_gripper and self._gripper_session_active
                and self._gripper_armed and ready and not self._gripper_faulted):
            self.cancel_prepare_exit(ctx)
            self.logger.info("SONIC退出：没有有效且健康的已校准夹爪会话，不执行收拢")
            return True
        if self._gripper_exit_complete:
            return True
        if self._gripper_exit_started_at is None:
            self._gripper_exit_started_at = time.monotonic()
            self._gripper_exit_settled_at = None
            for motion in self._gripper_motion_controllers.values():
                motion.pause()
            self._gripper_trigger_seen.clear()
            self._left_trigger_received_at = self._right_trigger_received_at = None
            self.logger.info(
                "SONIC正常退出：先低速收拢至本轮闭合软限位；暂停夹爪trigger；"
                "零力矩/制动/恢复/强制退出可立即中断；"
                f"speed={self._gripper_exit_speed}rad/s, timeout={self._gripper_exit_timeout}s")
        return False

    def cancel_prepare_exit(self, ctx: RobotControlContext) -> None:
        if self._gripper_exit_started_at is not None:
            self._gripper_trigger_seen.clear()
            self._left_trigger_received_at = self._right_trigger_received_at = None
            for motion in self._gripper_motion_controllers.values():
                motion.pause()
        self._gripper_exit_started_at = None
        self._gripper_exit_settled_at = None
        self._gripper_exit_complete = False

    def on_update(self, ctx: RobotControlContext, dt: float) -> None:
        # if ctx.is_orientation_unsafe(ctx.current_quat_xyzw):
        #     ctx.request_state(
        #         "com.bxi.basic_actions/zero_torque",
        #         trigger="sonic_orientation_safety",
        #     )
        #     return
        self._update_gripper(ctx, dt)
        if (self._gripper_faulted
                or (self.hardware_gripper and not self._gripper_session_active)):
            return
        self._apply_frame(ctx, self.sample_running_frame(ctx, dt, advance=True))

    def on_action(self, ctx: RobotControlContext, action_name: str) -> bool:
        if action_name != "reset_alignment":
            return False
        self.policy.reset_yaw_alignment()
        return True

    @staticmethod
    def _valid_trigger(value: object) -> Optional[float]:
        try:
            result = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(result):
            return None
        return float(np.clip(result, 0.0, 1.0))

    def _left_trigger_callback(self, msg: Float32) -> None:
        value = self._valid_trigger(msg.data)
        if (
            value is not None
            and self._gripper_session_active
            and self._gripper_motion_enabled
            and self._gripper_motion_ready
            and not self._gripper_faulted
            and self._gripper_exit_started_at is None
        ):
            with self._gripper_input_lock:
                self._left_trigger = value
                self._left_trigger_received_at = time.monotonic()
                self._gripper_trigger_seen.add("left")

    def _right_trigger_callback(self, msg: Float32) -> None:
        value = self._valid_trigger(msg.data)
        if (
            value is not None
            and self._gripper_session_active
            and self._gripper_motion_enabled
            and self._gripper_motion_ready
            and not self._gripper_faulted
            and self._gripper_exit_started_at is None
        ):
            with self._gripper_input_lock:
                self._right_trigger = value
                self._right_trigger_received_at = time.monotonic()
                self._gripper_trigger_seen.add("right")

    def _gripper_feedback_callback(self, msg) -> None:
        if (not self._gripper_session_active or self._gripper_faulted
                or (self._gripper_discovery_only and self._gripper_discovery_finished)):
            return
        bus = int(msg.bus)
        candidates = self._gripper_candidates.get(bus)
        if candidates is None:
            return
        response_id = int(msg.frame.can_id)
        frame_length = int(msg.frame.len)
        frame_flags = int(msg.frame.flags)
        data = [int(value) for value in msg.frame.data[: max(frame_length, 0)]]
        with self._gripper_feedback_lock:
            self._gripper_rx_counts[bus] += 1
            rx_count = self._gripper_rx_counts[bus]
            sent = set(self._gripper_sent_probes)
        if rx_count <= 3:
            self.logger.info(
                f"SONIC夹爪原始RX：bus={bus}, id=0x{response_id:X}, "
                f"flags={frame_flags}, len={frame_length}, data={data[:8]}"
            )
        if (not 0 <= response_id <= 0x7FF or frame_length != 8
                or frame_flags not in SUPPORTED_CAN_FRAME_FLAGS):
            return

        # Probe the configured DM identities on each physical port.  A strict
        # register reply selects and locks that port's complete profile for
        # this SONIC session.  Discovery replies are never decoded as MIT
        # state frames, including delayed replies that arrive after arming.
        matches = []
        for candidate in candidates:
            if (bus, candidate.protocol, candidate.can_id) not in sent:
                continue
            if (bus in self._gripper_auto_buses
                    and candidate.protocol is GripperProtocol.DM
                    and data[:4] == candidate.detection_payload
                    and struct.unpack("<I", bytes(data[4:8]))[0] == response_id
                    and response_id != candidate.master_id):
                with self._gripper_feedback_lock:
                    self._gripper_detection_errors[bus] = (
                        f"CAN{bus} DM身份不匹配：回复master_id={response_id}，"
                        f"配置={candidate.master_id}；不猜测其他协议、不使能")
                return
            resolved = candidate.resolve_detection_reply(
                response_id, data,
                auto_master_id=self._gripper_dm_auto_master_id,
            )
            if resolved is not None:
                # Discovery may observe FD replies from a motor configured
                # differently. Do not silently change the transport to match.
                if frame_flags != candidate.frame_flags:
                    with self._gripper_feedback_lock:
                        self._gripper_detection_errors[bus] = (
                            f"CAN{bus}回复flags={frame_flags}与配置"
                            f"flags={candidate.frame_flags}不同；需核对波特率/驱动"
                        )
                    return
                matches.append(resolved)
        if matches:
            first_reply = False
            selected_profile = matches[0]
            with self._gripper_feedback_lock:
                previous = self._gripper_profiles.get(bus)
                if len(matches) != 1:
                    self._gripper_detection_errors[bus] = (
                        f"CAN{bus}同一回复匹配多个协议"
                    )
                elif previous is None:
                    self._gripper_profiles[bus] = selected_profile
                    self._detected_gripper_buses.add(bus)
                    first_reply = True
                elif previous != selected_profile:
                    self._gripper_detection_errors[bus] = (
                        f"CAN{bus}发现多个身份："
                        f"{previous.protocol.value}/ID{previous.can_id}/MST{previous.master_id} 与 "
                        f"{selected_profile.protocol.value}/ID{selected_profile.can_id}/MST{selected_profile.master_id}"
                    )
            if first_reply:
                side = "左" if bus == self._left_bus else "右"
                self.logger.info(
                    f"SONIC{side}夹爪自动协议检测成功："
                    f"protocol={selected_profile.protocol.value}, bus={bus}, "
                    f"can_id={selected_profile.can_id}, "
                    f"master_id=0x{selected_profile.master_id:X}, "
                    f"flags={selected_profile.frame_flags}"
                )
            return

        # Motion entry re-reads ESC_ID, CTRL_MODE and all three codec ranges.
        # Only a solicited, identity/flags-matching response can satisfy a gate.
        with self._gripper_feedback_lock:
            profile = self._gripper_profiles.get(bus)
        if (profile is not None and profile.protocol is GripperProtocol.DM
                and response_id == profile.master_id and len(data) == 8
                and (data[0] | (data[1] << 8)) == profile.can_id and data[2] == 0x33):
            rid = data[3]
            expected = self._dm_register_expectations(profile)
            with self._gripper_feedback_lock:
                requested = (bus, rid) in self._gripper_register_requested
            if rid in expected and requested:
                value = struct.unpack("<I" if rid in (0x08, 0x0A) else "<f", bytes(data[4:8]))[0]
                valid = (frame_flags == profile.frame_flags and math.isfinite(value)
                         and math.isclose(value, expected[rid], rel_tol=1e-5, abs_tol=1e-6))
                with self._gripper_feedback_lock:
                    if valid:
                        self._gripper_register_values[(bus, rid)] = value
                    else:
                        self._gripper_feedback_errors[bus] = (
                            f"CAN{bus} DM寄存器0x{rid:02X}回读不符：value={value}, expected={expected[rid]}, flags={frame_flags}")
            return

        # DM parameter replies (including a mismatched RID/MST_ID or a delayed
        # read response) must not be interpreted as MIT position/torque data.
        dm_ids = {p.can_id for p in candidates if p.protocol is GripperProtocol.DM
                  and (profile is None or profile.protocol is GripperProtocol.DM)}
        if dm_ids:
            if (len(data) == 8 and (data[0] | (data[1] << 8)) in dm_ids
                    and data[2] in (0x33, 0x55, 0xAA)):
                return

        with self._gripper_feedback_lock:
            profile = self._gripper_profiles.get(bus)
        if (
            not self._gripper_armed
            or profile is None
            or response_id != profile.master_id
        ):
            return
        if frame_flags != profile.frame_flags:
            with self._gripper_feedback_lock:
                self._gripper_feedback_errors.setdefault(bus, f"CAN{bus} MIT feedback flags mismatch: flags={frame_flags}, data={data}")
            return
        try:
            if frame_length != 8:
                raise ValueError(f"expected 8 feedback bytes, got {frame_length}")
            feedback = profile.decode_feedback(data, received_at=time.monotonic())
            if feedback.motor_id != profile.can_id:
                return
        except (TypeError, ValueError) as exc:
            if bus not in self._bad_gripper_feedback_warned:
                self.logger.warning(f"忽略非法夹爪响应帧：{exc}")
                self._bad_gripper_feedback_warned.add(bus)
            return
        with self._gripper_feedback_lock:
            previous = self._gripper_feedback.get(bus)
            if (
                self._gripper_motion_enabled
                and previous is not None
                and abs(feedback.position - previous.position)
                > self._gripper_feedback_jump_limit_rad
            ):
                self._gripper_feedback_errors[bus] = (
                    f"CAN{bus}反馈位置单帧跳变："
                    f"{previous.position:.3f}->{feedback.position:.3f} rad"
                )
            self._gripper_feedback[bus] = feedback

    def _start_gripper_session(self, now: float) -> None:
        if not self.hardware_gripper:
            return
        self._log_dm_gripper_settings()
        self._reset_gripper_session(now)

    def _log_dm_gripper_settings(self):
        self.logger.info(
            f"SONIC夹爪软件版本={GRIPPER_RELEASE}；DM参数："
            f"pico_speed={self._gripper_target_speed_rad_s}rad/s, "
            f"pico_acceleration={self._gripper_grip_config['acceleration']}rad/s², "
            f"motion_feedback_speed_guard={self._gripper_motion_velocity_limit}rad/s, "
            f"motion={self._gripper_motion_enabled}, auto_calibrate={self._gripper_auto_calibrate}, "
            f"open_directions={self._gripper_open_directions}, "
            f"feedforward_max={self._gripper_calibration_kwargs['feedforward_max_nm']}Nm, "
            f"calibration_PD_FF_budget={self._gripper_calibration_kwargs['command_torque_limit_nm']}Nm, "
            f"motion_PD_FF_budget={self._gripper_grip_config['command_nm']}Nm, "
            f"hold_budget={self._gripper_grip_config['hold_nm']}Nm, "
            f"soft_feedback={self._gripper_grip_config['soft_nm']}Nm, "
            f"hard_feedback={self._gripper_motion_torque_limit_nm}Nm（估算，不是硬件限流）"
        )
    def _reset_gripper_session(self, now):
        self._left_trigger = self._right_trigger = 0.0
        self.cancel_prepare_exit(None)
        self._gripper_armed = False
        self._gripper_calibrated = False
        self._gripper_faulted = False
        self._gripper_fault_state_requested = False
        self._gripper_motion_ready = False
        self._gripper_motion_started_at = None
        self._gripper_discovery_finished = False
        self._bad_gripper_feedback_warned.clear()
        self._last_gripper_enable_time = None
        self._gripper_detection_started_at = float(now)
        self._last_gripper_detection_time = None
        self._left_trigger_received_at = None
        self._right_trigger_received_at = None
        self._gripper_trigger_seen.clear()
        self._gripper_trigger_stale_logged.clear()
        self._gripper_target_positions.clear()
        self._gripper_motion_feedback_at.clear()
        self._gripper_motion_controllers.clear()
        self._gripper_recent_motion.clear()
        self._gripper_recent_motion_at = -math.inf
        self._gripper_last_feedback_position.clear()
        self._gripper_feedback_stall_since.clear()
        self._gripper_pose_paused = False
        with self._gripper_feedback_lock:
            self._gripper_detection_cursor = {bus: 0 for bus in self._gripper_candidates}
            self._gripper_sent_probes.clear()
            self._gripper_rx_counts = {bus: 0 for bus in self._gripper_candidates}
            self._detected_gripper_buses.clear()
            self._gripper_detection_errors.clear()
            self._gripper_profiles.clear()
            self._gripper_feedback.clear()
            self._gripper_feedback_errors.clear()
        self._gripper_calibrators.clear()
        self._gripper_phase_snapshot.clear()
        self._gripper_diagnostic_at.clear()
        self._gripper_register_values.clear()
        self._gripper_register_requested.clear()
        self._gripper_register_started_at = self._gripper_register_last_tx = None
        self._gripper_session_active = True
    def _publish_gripper_detection(self, now: float) -> None:
        """At most one register-read candidate per bus per interval."""

        if (not self._gripper_session_active or self._gripper_faulted
                or self._gripper_discovery_finished or self._gripper_armed):
            return

        with self._gripper_feedback_lock:
            detected = set(self._detected_gripper_buses)
        for bus, candidates in self._gripper_candidates.items():
            # During scanning, keep probing after the first reply to detect
            # multiple identities within the configured candidate range.
            if self._gripper_dm_scan or bus not in detected:
                with self._gripper_feedback_lock:
                    cursor = self._gripper_detection_cursor[bus]
                    profile = candidates[cursor % len(candidates)]
                    self._gripper_sent_probes.add((bus, profile.protocol, profile.can_id))
                    self._gripper_detection_cursor[bus] = cursor + 1
                packet = profile.build_detection_packet()
                self._gripper_publisher.publish(packet)
        self._last_gripper_detection_time = float(now)

    def _publish_gripper_enable(self, now: float) -> None:
        if self._gripper_discovery_only and not self._gripper_enable_on_discovery:
            return
        # Latch before the first write, so a partial two-side enable is cleaned
        # up by the failure path if the second publish raises.
        self._gripper_armed = True
        for profile in tuple(self._gripper_profiles.values()):
            self._gripper_publisher.publish(
                DMMotor.build_motor_packet(
                    profile.bus,
                    profile.can_id,
                    DMMotor.enter_motor_mode(),
                    frame_flags=profile.frame_flags,
                )
            )
        self._last_gripper_enable_time = now

    @staticmethod
    def _dm_register_expectations(profile: GripperMotorProfile) -> dict:
        return {0x08: profile.can_id, 0x0A: 1,
                0x15: profile.position_max_rad, 0x16: profile.velocity_max_rad_s,
                0x17: profile.torque_max_nm}

    def _verify_gripper_registers(self, now: float, ctx) -> bool:
        with self._gripper_feedback_lock:
            profiles = dict(self._gripper_profiles)
            received = dict(self._gripper_register_values)
        missing = {
            bus: [rid for rid in self._dm_register_expectations(p) if (bus, rid) not in received]
            for bus, p in profiles.items() if p.protocol is GripperProtocol.DM
        }
        if not any(missing.values()):
            return True
        if self._gripper_register_started_at is None:
            self._gripper_register_started_at = now
        deadline = max(2.0, 6 * self.gripper_detection_interval_s
                       + self._gripper_calibration_kwargs["response_timeout_s"])
        if now - self._gripper_register_started_at >= deadline:
            self._fail_gripper_session(f"DM运动前只读寄存器核验超时，缺少{missing}；未发送使能/运动", ctx)
            return False
        if (self._gripper_register_last_tx is None
                or now - self._gripper_register_last_tx >= self.gripper_detection_interval_s):
            for bus, rids in missing.items():
                if not rids:
                    continue
                rid, profile = rids[0], profiles[bus]
                with self._gripper_feedback_lock:
                    self._gripper_register_requested.add((bus, rid))
                self._gripper_publisher.publish(DMMotor.build_motor_packet(
                    bus, 0x7FF, [profile.can_id, 0, 0x33, rid], frame_flags=profile.frame_flags))
            self._gripper_register_last_tx = now
        return False

    def _refresh_gripper_enable(
        self,
        now: float,
        *,
        interval_s: float | None = None,
    ) -> None:
        last_enable_time = self._last_gripper_enable_time
        interval = (
            self.gripper_enable_interval_s
            if interval_s is None
            else min(self.gripper_enable_interval_s, float(interval_s))
        )
        if (
            last_enable_time is None
            or now - last_enable_time >= interval
        ):
            self._publish_gripper_enable(now)

    def _disable_grippers(self, *, repeats: int = 1) -> None:
        if self._gripper_discovery_only and not self._gripper_enable_on_discovery:
            return
        errors = []
        for _ in range(max(int(repeats), 1)):
            for profile in tuple(self._gripper_profiles.values()):
                try:
                    self._gripper_publisher.publish(
                        DMMotor.build_motor_packet(
                            profile.bus, profile.can_id, DMMotor.exit_motor_mode(),
                            frame_flags=profile.frame_flags,
                        )
                    )
                except Exception as exc:
                    errors.append(f"CAN{profile.bus}: {exc}")
        if errors:
            self.logger.error("夹爪失能帧发布失败，不能确认电机已停止；请使用现场急停：" + "; ".join(errors))

    def _publish_gripper_target(
        self,
        bus: int,
        target_position: float,
        *,
        kp: float,
        kd: float,
        t_ff: float = 0.0,
        v_des: float = 0.0,
    ) -> None:
        if (self._gripper_discovery_only or self._gripper_faulted
                or not self._gripper_session_active or not self._gripper_armed):
            return
        profile = self._gripper_profiles[bus]
        if not (math.isfinite(target_position) and profile.position_range[0] < target_position < profile.position_range[1]
                and math.isfinite(kp) and 0 < kp <= 500
                and math.isfinite(kd) and 0 < kd <= 5
                and math.isfinite(v_des) and profile.velocity_range[0] < v_des < profile.velocity_range[1]
                and math.isfinite(t_ff) and profile.torque_range[0] < t_ff < profile.torque_range[1]):
            raise ValueError("refusing invalid/out-of-range MIT command")
        if profile.protocol is not GripperProtocol.DM:
            raise ValueError('this controller supports DM grippers only')
        command = JointControl(
            p_des=float(target_position),
            v_des=float(v_des),
            kp=float(kp),
            kd=float(kd),
            t_ff=float(t_ff),
        )
        data = DMMotor.pack_cmd(
            command,
            p_range=profile.position_range,
            v_range=profile.velocity_range,
            t_range=profile.torque_range,
            kp_range=(0.0, 500.0),
            kd_range=(0.0, 5.0),
        )
        self._gripper_publisher.publish(
            DMMotor.build_motor_packet(
                bus,
                profile.can_id,
                data,
                frame_flags=profile.frame_flags,
            )
        )
        self._collection_call("sent", bus, data, profile)
    @staticmethod
    def _slew_target(current: float, desired: float, speed: float, dt: float) -> float:
        if not all(math.isfinite(v) for v in (current, desired, speed, dt)) or dt < 0:
            raise ValueError("invalid gripper slew input")
        step = max(float(speed) * min(float(dt), 0.05), 0.0)
        if abs(desired - current) <= step:
            return float(desired)
        return float(current + math.copysign(step, desired - current))

    def _trigger_is_fresh(self, side: str, now: float, snapshot=None) -> bool:
        if snapshot is None:
            with self._gripper_input_lock:
                received_at = getattr(self, f'_{side}_trigger_received_at')
                seen = side in self._gripper_trigger_seen
        else:
            _, received_at, seen = snapshot
        side_label = "左" if side == "left" else "右"
        if not seen or received_at is None:
            if side not in self._gripper_trigger_stale_logged:
                self._motion_log('info',
                    f"SONIC{side_label}夹爪等待新trigger，保持当前位置"
                )
                self._gripper_trigger_stale_logged.add(side)
            return False
        if not 0 <= now - received_at <= self._gripper_trigger_timeout_s:
            if side not in self._gripper_trigger_stale_logged:
                self._motion_log('info',
                    f"SONIC{side_label}夹爪trigger流已过期，保持当前位置"
                )
                self._gripper_trigger_stale_logged.add(side)
            return False
        self._gripper_trigger_stale_logged.discard(side)
        return True

    def _live_pose_allows_gripper_motion(self) -> bool:
        if not self._gripper_pause_with_live_reference:
            return True
        # PICO A+X can explicitly latch the body reference while the source
        # stream is still publishing fresh frames.  Freshness alone therefore
        # is not enough to decide whether the body is actively teleoperated.
        # Keep the gripper at its last MIT target during both the explicit
        # HOLD and the guarded re-arm transition.  ``last_status`` is part of
        # the existing SonicPolicy status contract; older policy objects that
        # do not expose it simply fall through to the freshness check below.
        reference_status = getattr(self.policy, "last_status", None)
        if reference_status in ("held_reference", "rearming_reference"):
            if not self._gripper_pose_paused:
                self._motion_log('info',
                    "SONIC PICO全身POSE处于显式暂停/恢复阶段；DM夹爪同步保持；暂停训练采样"
                )
            self._gripper_pose_paused = True
            self._clear_gripper_inputs()
            return False
        try:
            fresh = bool(
                self.policy.has_fresh_live_reference(
                    self.live_reference_timeout_s
                )
            )
        except Exception as exc:
            if not self._gripper_pose_paused:
                self._motion_log('warning',
                    f"SONIC无法确认PICO POSE新鲜度；DM夹爪保持；暂停训练采样：{exc}"
                )
            fresh = False
        if not fresh:
            if not self._gripper_pose_paused:
                self._motion_log('info',
                    "SONIC PICO全身POSE已暂停或过期；DM夹爪同步保持；暂停训练采样"
                )
            self._gripper_pose_paused = True
            self._clear_gripper_inputs()
            return False
        if self._gripper_pose_paused:
            self._motion_log('info',
                "SONIC PICO全身POSE已恢复；DM夹爪等待新trigger后限速恢复"
            )
            self._gripper_pose_paused = False
        return True

    def _motion_settings(self, bus=None):
        settings = GripMotionSettings(**self._gripper_grip_config,
            kp=self._gripper_kp, kd=self._gripper_kd,
            stall_s=self._gripper_motion_stall_timeout_s,
            tracking_error=self._gripper_calibration_kwargs['tracking_error_rad'],
            stopped_velocity=self._gripper_calibration_kwargs['stopped_velocity_rad_s'],
            ff_max=self._gripper_calibration_kwargs['feedforward_max_nm'],
            ff_slew=self._gripper_calibration_kwargs['feedforward_slew_nm_s'])
        return settings

    def _cancel_blocked_gripper_exit(self, ctx, reason):
        self.logger.warning("SONIC正常退出收拢未完成：" + reason +
            "；已取消本次normal请求，保留受限保持，不松爪、不自动零力矩；"
            "当前仍为SONIC，可用新鲜PICO打开指令解除受阻保持，或显式安全退出")
        # Requesting the current state cancels the pending normal gate without
        # on_exit/disable. No framework API change or uncalibrated re-entry.
        self.cancel_prepare_exit(ctx)
        if ctx is not None:
            ctx.request_state("com.bxi.sonic/sonic_teleop", trigger="gripper_exit_blocked")


    def _update_gripper_motion(
        self,
        ctx: RobotControlContext | None,
        now: float,
        dt: float,
    ) -> None:
        """DM calibrated/fixed-limit pair tracking and close-before-exit."""

        self._gripper_control_snapshot = None
        exiting = self._gripper_exit_started_at is not None
        pose_allows_motion = False if exiting else self._live_pose_allows_gripper_motion()
        # Immutable pair + time sampled AFTER callbacks are excluded. A message
        # arriving after this snapshot is deliberately used next cycle.
        with self._gripper_feedback_lock, self._gripper_input_lock:
            feedback = dict(self._gripper_feedback)
            feedback_errors = dict(self._gripper_feedback_errors)
            inputs = {side: (getattr(self, f'_{side}_trigger'),
                             getattr(self, f'_{side}_trigger_received_at'),
                             side in self._gripper_trigger_seen) for side in ('left', 'right')}
            now = time.monotonic()
        for bus, reason in feedback_errors.items():
            self._fail_gripper_session(reason, ctx)
            return
        selected = dict(self._gripper_profiles)

        if now-self._gripper_recent_motion_at >= .02:
            self._gripper_recent_motion_at = now
            self._gripper_recent_motion.append(dict(t=now, samples={bus: dict(
                p=f.position, v=f.velocity, tau=f.torque, received_at=f.received_at,
                target=self._gripper_target_positions.get(bus),
                mode=(self._gripper_motion_controllers[bus].mode if bus in self._gripper_motion_controllers else 'handoff'),
                ff=(self._gripper_motion_controllers[bus].command_feedforward if bus in self._gripper_motion_controllers else 0.),
                command_velocity=(self._gripper_motion_controllers[bus].command_velocity if bus in self._gripper_motion_controllers else 0.),
                command_kp=(self._gripper_motion_controllers[bus].command_kp if bus in self._gripper_motion_controllers else None),
                command_kd=(self._gripper_motion_controllers[bus].command_kd if bus in self._gripper_motion_controllers else None))
                for bus, f in feedback.items()}))

        response_timeout = float(self._gripper_calibration_kwargs["response_timeout_s"])
        feedback_timeout = float(self._gripper_calibration_kwargs["feedback_timeout_s"])
        temperature_limit_mos = int(
            self._gripper_calibration_kwargs["maximum_mos_temperature_c"]
        )
        temperature_limit_motor = int(
            self._gripper_calibration_kwargs["maximum_motor_temperature_c"]
        )
        velocity_limit = self._gripper_motion_velocity_limit
        if self._gripper_motion_started_at is None:
            self._gripper_motion_started_at = now
        def feedback_error(bus, profile, item):
            side = "left" if bus == self._left_bus else "right"
            side_label = "左" if side == "left" else "右"
            if not all(math.isfinite(v) for v in (item.position, item.velocity, item.torque,
                       item.mos_temperature_c, item.motor_temperature_c, item.received_at)):
                return f"{side_label}夹爪反馈含非有限值"
            if item.received_at > now + 0.05:
                return f"{side_label}夹爪反馈时间戳异常"
            if item.received_at < self._gripper_motion_feedback_at.get(bus, -math.inf):
                return f"CAN{bus}反馈时间倒退"
            if now - item.received_at > feedback_timeout:
                return f"{side_label}夹爪反馈超时（>{feedback_timeout:.2f}s）"
            if item.status is not None and item.status != 0x1:
                return f"{side_label}夹爪DM状态异常：0x{item.status:X}"
            if item.mos_temperature_c >= temperature_limit_mos:
                return f"{side_label}夹爪MOS温度过高：{item.mos_temperature_c:.1f} C"
            if item.motor_temperature_c >= temperature_limit_motor:
                return f"{side_label}夹爪电机温度过高：{item.motor_temperature_c:.1f} C"
            if abs(item.velocity) >= velocity_limit:
                return f"{side_label}夹爪反馈速度异常：{item.velocity:.3f} rad/s"
            if abs(item.torque) >= self._gripper_motion_torque_limit_nm:
                return f"{side_label}夹爪反馈力矩超过运动保护阈值：{item.torque:.3f} Nm"
            open_position, closed_position = self._soft_limits_for(bus, profile)
            # Allow one small feedback/settling tolerance, never extend targets.
            margin = min(0.02, self._gripper_calibration_kwargs["limit_margin_rad"] * 0.25)
            if not (min(open_position, closed_position) - margin <= item.position <= max(open_position, closed_position) + margin):
                return f"{side_label}夹爪反馈位置{item.position:.3f}超出已配置软限位"
            return None

        missing = []
        for bus, profile in tuple(selected.items()):
            item = feedback.get(bus)
            if item is None:
                missing.append(bus)
                continue
            reason = feedback_error(bus, profile, item)
            if reason:
                self._fail_gripper_session(reason, ctx)
                return
                selected.pop(bus)

        if missing:
            if now - self._gripper_motion_started_at >= response_timeout:
                self._fail_gripper_session(f"夹爪使能后缺少有效MIT反馈：CAN{missing}", ctx)
            else:
                self._refresh_gripper_enable(now, interval_s=max(response_timeout * 0.5, 0.05))
            return
        self._refresh_gripper_enable(now)
        if not selected:
            return
        # A side fault may have cancelled normal while validating this cycle.
        exiting = self._gripper_exit_started_at is not None
        if not self._gripper_motion_ready:
            self._gripper_motion_controllers.clear()
            for bus, item in feedback.items():
                self._gripper_target_positions[bus] = float(item.position)
                self._gripper_last_feedback_position[bus] = float(item.position)
                self._gripper_feedback_stall_since[bus] = None
            self._gripper_motion_ready = True
            self._gripper_calibrated = True
            self._motion_log('info',
                "SONIC左右DM夹爪反馈健康，已锁定当前位置；"
                "MIT位置运动已就绪，等待新的左右PICO trigger"
            )

        exit_timed_out = (exiting and not self._gripper_exit_complete
                         and now-self._gripper_exit_started_at >= self._gripper_exit_timeout)
        motion_speed = self._gripper_exit_speed if exiting else self._gripper_target_speed_rad_s
        exit_settled = exiting
        all_fresh = True
        # Keep command effort below the measured-torque trip threshold. This
        # headroom does not replace the feedback guard or hardware protection.
        command_budget = self._gripper_grip_config['command_nm']
        planned = {}
        control_samples = {}
        for bus, trigger, side in (
            (self._left_bus, inputs['left'][0], "left"),
            (self._right_bus, inputs['right'][0], "right"),
        ):
            if bus not in selected:
                continue
            target = self._gripper_target_positions[bus]
            closing = exiting
            active = (not self._gripper_exit_complete if exiting else
                      pose_allows_motion and self._trigger_is_fresh(side, now, inputs[side]))
            new_feedback = feedback[bus].received_at > self._gripper_motion_feedback_at.get(bus, -math.inf)
            all_fresh = all_fresh and new_feedback
            item = feedback[bus]
            open_position, closed_position = self._soft_limits_for(bus, self._gripper_profiles[bus])
            desired = (closed_position if closing else closed_position+(1.0-trigger)*(open_position-closed_position)) if active else target
            motion = self._gripper_motion_controllers.get(bus)
            if motion is None:
                motion = GripMotionController(self._motion_settings(bus), target)
                self._gripper_motion_controllers[bus] = motion
            elif motion.command_position != target:
                motion.target = target
                motion.command_position = target
                motion.pause()
            acceleration = self._gripper_exit_acceleration if exiting else self._gripper_grip_config['acceleration']
            braking = self._gripper_exit_acceleration if exiting else self._gripper_grip_config['braking_acceleration']
            side_speed = motion_speed
            if (motion.settings.acceleration != acceleration or motion.settings.braking_acceleration != braking):
                motion.settings = replace(motion.settings, acceleration=acceleration, braking_acceleration=braking)
            if exit_timed_out and not motion.locally_held:
                motion.block(item, closed_position, 'exit_timeout')
            before_mode = motion.mode
            target, torque_ff = motion.update(item, now, dt, desired=desired,
                open_position=open_position, closed_position=closed_position,
                speed=side_speed, active=active and not exit_timed_out, fast_follow=not closing)
            if before_mode != motion.mode:
                self._motion_log('info', f"SONIC夹爪CAN{bus}局部控制：{before_mode} -> {motion.mode}；"
                    f"reason={motion.reason}, target={target:.4f}, torque={item.torque:.3f}Nm；未请求整机零力矩")
            planned[bus] = (target, torque_ff, motion.command_velocity, motion.command_kp, motion.command_kd)
            control_samples[bus] = dict(active=active and not exit_timed_out, mode=motion.mode,
                effective_speed_limit=motion.effective_speed, rate_reason=motion.rate_reason)
            timing = motion.response_timing.observe(item, now, desired, active, motion.mode,
                None if exiting else inputs[side][1])
            if timing is not None:
                self._motion_log('info', "SONIC夹爪实测响应：", dict(bus=bus,
                    phase='exit_closing' if exiting else 'pico_motion', **timing))
            if closing:
                exit_settled = exit_settled and (
                    not motion.locally_held and torque_ff == 0.0
                    and
                    abs(motion.target - closed_position) <= 1e-6
                    and abs(item.position - closed_position) <= 0.01
                    and abs(item.velocity) <= self._gripper_calibration_kwargs["stopped_velocity_rad_s"])
            # Completion timing above uses every fresh feedback. Keep verbose
            # JSON at 2 Hz to avoid loading the body-control loop with logging.
            if now - self._gripper_diagnostic_at.get(bus, -math.inf) >= .5:
                self._gripper_diagnostic_at[bus] = now
                self._motion_log('info', "SONIC夹爪运动诊断：", dict(
                    bus=bus, phase="exit_closing" if exiting else "pico_motion", input_active=active,
                    position=item.position, target=target, velocity=item.velocity,
                    desired=desired, trigger=trigger, command_velocity=motion.command_velocity,
                    trajectory_velocity=motion.velocity, mode=motion.mode,
                    reference_target=motion.target, command_kp=motion.command_kp, command_kd=motion.command_kd,
                    effective_speed_limit=motion.effective_speed,
                    rate_reason=motion.rate_reason,
                    tracking_error=desired-item.position, feedback_age_s=now-item.received_at,
                    input_received_at=inputs[side][1],
                    monotonic_s=now,
                    torque=item.torque, t_ff=torque_ff,
                    estimated_total_torque=motion.command_kp * (target-item.position)
                        + motion.command_kd * (motion.command_velocity-item.velocity) + torque_ff,
                    command_budget=motion.settings.command_nm,
                    protocol=self._gripper_profiles[bus].protocol.value), key=('motion', bus))
        for bus, (target, torque_ff, command_velocity, command_kp, command_kd) in planned.items():
            self._gripper_target_positions[bus] = target
            self._gripper_motion_feedback_at[bus] = feedback[bus].received_at
            self._publish_gripper_target(
                bus,
                target,
                kp=command_kp,
                kd=command_kd,
                t_ff=torque_ff,
                v_des=command_velocity,
            )
        if set(control_samples) == set(self._gripper_profiles):
            self._gripper_control_snapshot = dict(monotonic=now, feedback=feedback, inputs=inputs,
                sides=control_samples, exiting=exiting, pose_allowed=pose_allows_motion,
                policy_status=getattr(self.policy, 'last_status', None))
        if exiting and (exit_timed_out or any(m.locally_held for m in self._gripper_motion_controllers.values())):
            self._cancel_blocked_gripper_exit(ctx, '收拢超时' if exit_timed_out else '夹持/受阻，尚未到闭合软限位')
            return
        # Cached samples cannot establish a settled dwell or complete an exit.
        if exiting and not self._gripper_exit_complete:
            if not exit_settled:
                self._gripper_exit_settled_at = None
            elif all_fresh:
                if self._gripper_exit_settled_at is None:
                    self._gripper_exit_settled_at = now
                elif now - self._gripper_exit_settled_at >= 0.5:
                    self._gripper_exit_complete = True
                    for motion in self._gripper_motion_controllers.values():
                        motion.pause()
                    if set(selected) != set(self._gripper_profiles):
                        self.logger.info(f"SONIC正常退出：可用侧CAN{sorted(selected)}已闭合稳定；"
                                         f"未参与侧CAN{sorted(set(self._gripper_profiles)-set(selected))}未确认闭合，不驱动；允许切换")
                    else:
                        self.logger.info("SONIC正常退出：左右夹爪已到闭合软限位并稳定，允许切换；退出时发送失能")


    def _log_gripper_phase_change(
        self,
        bus: int,
        calibrator: GripperCalibrator,
    ) -> None:
        previous = self._gripper_phase_snapshot[bus]
        if calibrator.phase is previous:
            return
        self._gripper_phase_snapshot[bus] = calibrator.phase
        side = "左" if bus == self._left_bus else "右"
        labels = {
            CalibrationPhase.SETTLING: "收到响应，正在稳定当前位置",
            CalibrationPhase.SEEKING_OPEN: "开始低速寻找张开限位",
            CalibrationPhase.RELEASING_INITIAL: "初始接近张开端点；卸载后限程退让，再重新验证端点（仅一次）",
            CalibrationPhase.BACKING_OFF_OPEN: "已检测张开限位，正在回退",
            CalibrationPhase.SEEKING_CLOSED: "开始低速寻找闭合限位",
            CalibrationPhase.BACKING_OFF_CLOSED: "已检测闭合限位，正在回退",
            CalibrationPhase.RETURNING_OPEN: "正在低速返回张开位置",
            CalibrationPhase.READY: "校准完成",
        }
        message = labels.get(calibrator.phase)
        if message is not None:
            self.logger.info(f"SONIC{side}夹爪：{message}；{json.dumps(calibrator.diagnostics(), ensure_ascii=False)}")

    def _fail_gripper_session(
        self,
        reason: str,
        ctx: RobotControlContext | None = None,
    ) -> None:
        if self._gripper_faulted:
            return
        self._gripper_faulted = True
        self._gripper_calibrated = False
        self._gripper_motion_ready = False
        self._gripper_trigger_seen.clear()
        self._gripper_motion_controllers.clear()
        self._left_trigger_received_at = self._right_trigger_received_at = None
        if self._gripper_armed:
            self._disable_grippers(repeats=3)
            self._gripper_armed = False
            suffix = "；已请求左右夹爪失能（3轮），实际失能需反馈确认"
        else:
            suffix = "；检测阶段未发送使能、失能或运动帧"
        fault_feedback = {bus: vars(f) for bus, f in dict(self._gripper_feedback).items()}
        fault_history = list(self._gripper_recent_motion)
        self.logger.error(f"SONIC夹爪失败：{reason}{suffix}")
        if ctx is not None and not self._gripper_fault_state_requested:
            request_state = getattr(ctx, "request_state", None)
            if callable(request_state):
                self._gripper_fault_state_requested = True
                request_state(
                    "com.bxi.basic_actions/zero_torque",
                    trigger="gripper_fault",
                )
        # Safety request precedes potentially expensive history serialization.
        self.logger.error("SONIC夹爪故障时反馈：" + json.dumps(fault_feedback, ensure_ascii=False))
        if fault_history:
            self.logger.error("SONIC夹爪故障前运动记录：" + json.dumps(fault_history, ensure_ascii=False))

    def _update_gripper(
        self,
        ctx_or_dt: RobotControlContext | float | None,
        dt: float | None = None,
    ) -> None:
        try:
            self._update_gripper_impl(ctx_or_dt, dt)
        except Exception as exc:
            self._fail_gripper_session(
                f"夹爪控制异常 {type(exc).__name__}: {exc}",
                None if dt is None else ctx_or_dt,
            )

    def _update_gripper_impl(
        self,
        ctx_or_dt: RobotControlContext | float | None,
        dt: float | None = None,
    ) -> None:
        ctx = None if dt is None else ctx_or_dt
        dt = float(ctx_or_dt) if dt is None else float(dt)
        if not self.hardware_gripper or not self._gripper_session_active:
            return
        if self._gripper_faulted:
            return
        def fail(reason: str) -> None:
            self._fail_gripper_session(reason, ctx)

        if not math.isfinite(dt) or dt < 0:
            fail("invalid gripper control dt")
            return
        with self._gripper_feedback_lock:
            errors = dict(self._gripper_feedback_errors)
            detection_errors = dict(self._gripper_detection_errors)
        # Also enforce conflicts arriving AFTER arming; never switch a live
        # session to a new protocol or ignore a late second identity.
        if detection_errors:
            fail("协议身份异常：" + ", ".join(detection_errors.values()))
            return
        for bus, reason in errors.items():
            self._fail_gripper_session(reason, ctx)
            return
        if (self._gripper_armed and self._gripper_auto_calibrate and not self._gripper_motion_ready
                and self._gripper_pause_with_live_reference
                and getattr(self.policy, "last_status", None) in
                ("held_reference", "rearming_reference", "stale_hold")):
            fail("夹爪校准期间收到PICO暂停/断流状态；本次校准已取消，需重新进入SONIC")
            return

        if (
            self._gripper_motion_enabled
            and self._gripper_armed
            and self._gripper_profiles
            and (not self._gripper_auto_calibrate or self._gripper_motion_ready)
        ):
            self._update_gripper_motion(ctx, time.monotonic(), dt)
            return
        if self._gripper_discovery_finished:
            # In enable-only diagnostic mode the motors remain armed after a
            # successful read-only discovery.  Refresh the mode command at a
            # low rate so the green status LED remains a persistent, visible
            # indication; no calibration or MIT target is ever sent here.
            if self._gripper_armed:
                self._refresh_gripper_enable(time.monotonic())
            return
        now = time.monotonic()
        if not self._gripper_armed:
            with self._gripper_feedback_lock:
                detected = set(self._detected_gripper_buses)
                detection_errors = dict(self._gripper_detection_errors)
                selected_profiles = dict(self._gripper_profiles)
            if detection_errors:
                reason = ", ".join(
                    detection_errors[bus] for bus in sorted(detection_errors)
                )
                fail("自动协议检测歧义：" + reason)
                return
            expected = set(self._gripper_candidates)
            elapsed = now - self._gripper_detection_started_at
            if self._gripper_dm_scan and elapsed < self.gripper_detection_timeout_s:
                # Reserve the final interval for replies to the last request.
                last_probe = self._last_gripper_detection_time
                if (elapsed < self.gripper_detection_timeout_s - self.gripper_detection_interval_s
                        and (last_probe is None or
                             now - last_probe >= self.gripper_detection_interval_s)):
                    self._publish_gripper_detection(now)
                return
            if self._gripper_dm_scan:
                with self._gripper_feedback_lock:
                    cursors = dict(self._gripper_detection_cursor)
                    rx_counts = dict(self._gripper_rx_counts)
                self.logger.info(
                    f"SONIC夹爪只读扫描汇总：TX请求数={cursors}, ROS原始RX数={rx_counts}；"
                    "TX计数仅表示ROS已发布，不证明物理CAN发送成功"
                )
                if any(cursors[bus] < len(items)
                       for bus, items in self._gripper_candidates.items()):
                    fail("扫描窗口内未覆盖全部候选ID；拒绝使能")
                    return
            if detected != expected:
                if (
                    now - self._gripper_detection_started_at
                    >= self.gripper_detection_timeout_s
                ):
                    missing = []
                    for bus in sorted(expected - detected):
                        side = "左" if bus == self._left_bus else "右"
                        protocols = "|".join(
                            sorted({profile.protocol.value
                            for profile in self._gripper_candidates[bus]
                            })
                        )
                        missing.append(
                            f"{side}(protocols={protocols},bus={bus})"
                        )
                    fail(
                        "只读在线检测超时，缺少" + ", ".join(missing)
                    )
                    return
                last_probe = self._last_gripper_detection_time
                if (
                    last_probe is None
                    or now - last_probe >= self.gripper_detection_interval_s
                ):
                    self._publish_gripper_detection(now)
                return

            if set(selected_profiles) != expected:
                fail(
                    "自动协议检测内部状态不完整，拒绝使能"
                )
                return

            if self._gripper_motion_enabled and not self._verify_gripper_registers(now, ctx):
                return

            if self._gripper_motion_enabled and self._gripper_auto_calibrate:
                self.logger.info("SONIC夹爪运动前寄存器核验通过：" + str(self._gripper_register_values))
                self._gripper_profiles = selected_profiles
                self._create_gripper_calibrators(now)
                self._publish_gripper_enable(now)
                self._gripper_armed = True
                self._gripper_motion_started_at = now
                self.logger.info("SONIC DM夹爪身份已锁定，开始受保护的低速自动校准")
                return

            if self._gripper_motion_enabled:
                self._gripper_profiles = selected_profiles
                for bus, profile in self._gripper_profiles.items():
                    self._soft_limits_for(bus, profile)
                self._publish_gripper_enable(now)
                self._gripper_armed = True
                self._gripper_motion_started_at = now
                profiles = ", ".join(
                    f"{profile.label}:{profile.protocol.value}/CAN{profile.bus}/"
                    f"ID{profile.can_id}/MST{profile.master_id}/flags{profile.frame_flags}"
                    for profile in self._gripper_profiles.values()
                )
                self.logger.info(
                    "SONIC夹爪身份已锁定，已使能；等待左右DM反馈健康后发送MIT位置帧："
                    + profiles
                )
                return

            if self._gripper_discovery_only:
                self._gripper_profiles = selected_profiles
                self._gripper_discovery_finished = True
                if self._gripper_enable_on_discovery:
                    self._publish_gripper_enable(now)
                    self._gripper_armed = True
                    profiles = ", ".join(
                        f"{profile.label}:{profile.protocol.value}/CAN{profile.bus}/"
                        f"ID{profile.can_id}/MST{profile.master_id}/flags{profile.frame_flags}"
                        for profile in self._gripper_profiles.values()
                    )
                    self.logger.info(
                        "SONIC夹爪只读探测成功，已持续使能（未校准、未运动）："
                        + profiles
                        + "；DM协议无独立使能ACK，绿灯持续亮和CAN实际发送是物理确认"
                    )
                else:
                    self.logger.info(
                        "SONIC夹爪只读探测完成：身份已记录；未使能、未校准、未运动。"
                        "请先核对CTRL_MODE、PMAX/VMAX/TMAX和CAN波特率，再安排开合测试"
                    )
                return

            fail("拒绝没有显式运动授权的夹爪配置")
            return

        with self._gripper_feedback_lock:
            feedback = dict(self._gripper_feedback)

        waiting_buses = tuple(
            bus
            for bus, calibrator in self._gripper_calibrators.items()
            if calibrator.phase is CalibrationPhase.WAITING_FEEDBACK
        )
        if waiting_buses and not all(bus in feedback for bus in waiting_buses):
            for bus, calibrator in self._gripper_calibrators.items():
                if bus not in feedback:
                    calibrator.update(None, now, dt)
                else:
                    calibrator._check_common_faults(feedback[bus], now)
                if calibrator.failed:
                    fail(
                        calibrator.failure_reason or "unknown calibration error"
                    )
                    return
            self._refresh_gripper_enable(now, interval_s=max(
                self._gripper_calibration_kwargs["response_timeout_s"] * 0.5, 0.05))
            return

        planned = {}
        for bus, calibrator in self._gripper_calibrators.items():
            target = calibrator.update(feedback.get(bus), now, dt)
            self._log_gripper_phase_change(bus, calibrator)
            if now - self._gripper_diagnostic_at.get(bus, -math.inf) >= 0.5:
                self._gripper_diagnostic_at[bus] = now
                self.logger.info(f"SONIC夹爪校准诊断 CAN{bus}："
                                 + json.dumps(calibrator.diagnostics(), ensure_ascii=False))
            if calibrator.failed:
                fail(
                    calibrator.failure_reason or "unknown calibration error"
                )
                return
            if target is not None:
                planned[bus] = target

        self._refresh_gripper_enable(now)
        # READY sides keep receiving bounded holding frames while the partner
        # is still calibrating. No target is sent if either side failed above.
        for bus, target in planned.items():
            self._publish_gripper_target(bus, target, kp=self._gripper_calibration_kp,
                                         kd=self._gripper_calibration_kd,
                                         t_ff=self._gripper_calibrators[bus].feedforward_torque)

        if not all(
            calibrator.ready for calibrator in self._gripper_calibrators.values()
        ):
            return

        if not self._gripper_calibrated:
            self._gripper_calibrated = True
            details = []
            for bus, calibrator in self._gripper_calibrators.items():
                side = "左" if bus == self._left_bus else "右"
                details.append(
                    f"{side}[闭={calibrator.closed_position:.3f}, "
                    f"开={calibrator.open_position:.3f}]"
                )
            self.logger.info("SONIC夹爪校准完成，PICO trigger开始接管：" + ", ".join(details))

        if self._gripper_motion_enabled:
            with self._gripper_feedback_lock:
                current_feedback = dict(self._gripper_feedback)
            if set(current_feedback) != set(self._gripper_profiles):
                return
            for bus in current_feedback:
                # Preserve the stable calibration command across the handoff.
                # Do not replace it with a measured overshoot outside the soft
                # endpoint; READY intentionally allows small tracking error.
                self._gripper_target_positions[bus] = float(
                    self._gripper_calibrators[bus].target_position)
                self._gripper_feedback_stall_since[bus] = None
            self._gripper_motion_ready = True
            self.logger.info(
                "SONIC夹爪自动校准完成，已锁定软限位并等待新的trigger"
            )
            return

        fail("校准结束但运动授权不存在，拒绝未经门控的扳机输出")


__all__ = [
    "HEAD_JOINT_NAMES",
    "SONIC_HEAD_JOINTS",
    "SONIC_OUTPUT_JOINTS",
    "SonicTeleopState",
]
