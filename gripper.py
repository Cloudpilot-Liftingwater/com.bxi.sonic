#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import struct
from collections import deque
from dataclasses import dataclass, replace
from enum import Enum
from typing import Optional, Sequence
import communication.msg as bxiMsg


CAN_CLASSIC_FLAGS = 0x00
CAN_FD_FLAGS = 0x04
CAN_FD_BRS_FLAGS = 0x05
SUPPORTED_CAN_FRAME_FLAGS = frozenset(
    (CAN_CLASSIC_FLAGS, CAN_FD_FLAGS, CAN_FD_BRS_FLAGS)
)


class GripperProtocol(str, Enum):
    DM = "dm"


class GripperProtocolMode(str, Enum):
    AUTO = "auto"
    DM = "dm"


@dataclass
class JointControl:
    """Motor joint command data."""

    p_des: float = 0.0
    v_des: float = 0.0
    kp: float = 0.0
    kd: float = 0.0
    t_ff: float = 0.0


@dataclass(frozen=True)
class MotorFeedback:
    """Decoded MIT-mode motor feedback received from one physical CAN bus."""

    motor_id: int
    # DM feedback stores the motor state in the high nibble of D0.
    status: Optional[int]
    position: float
    velocity: float
    torque: float
    mos_temperature_c: float
    motor_temperature_c: float
    received_at: float


@dataclass(frozen=True)
class GripperMotorProfile:
    """Per-bus wire protocol and MIT scaling for one physical gripper."""

    label: str
    protocol: GripperProtocol | str
    bus: int
    can_id: int
    master_id: int
    frame_flags: int
    position_max_rad: float
    velocity_max_rad_s: float
    torque_max_nm: float

    def __post_init__(self) -> None:
        try:
            protocol = GripperProtocol(self.protocol)
        except ValueError as exc:
            raise ValueError("this controller supports DM grippers only") from exc
        object.__setattr__(self, "protocol", protocol)
        if not self.label:
            raise ValueError("gripper profile label must not be empty")
        if not 0 <= int(self.bus) <= 0xFF:
            raise ValueError("gripper bus must fit in uint8")
        if not 0 <= int(self.can_id) <= 0x0F:
            raise ValueError("gripper CAN ID must fit in the MIT feedback D0 ID")
        if not 0 <= int(self.master_id) <= 0x7FF:
            raise ValueError("gripper master ID must be an 11-bit CAN ID")
        if int(self.frame_flags) not in SUPPORTED_CAN_FRAME_FLAGS:
            raise ValueError(
                "gripper frame_flags must be 0 (classic CAN), "
                "4 (CAN-FD), or 5 (CAN-FD+BRS)"
            )
        limits = (
            float(self.position_max_rad),
            float(self.velocity_max_rad_s),
            float(self.torque_max_nm),
        )
        if not all(math.isfinite(value) and value > 0.0 for value in limits):
            raise ValueError("gripper MIT ranges must be finite and positive")

    @property
    def position_range(self) -> tuple[float, float]:
        return (-self.position_max_rad, self.position_max_rad)

    @property
    def velocity_range(self) -> tuple[float, float]:
        return (-self.velocity_max_rad_s, self.velocity_max_rad_s)

    @property
    def torque_range(self) -> tuple[float, float]:
        return (-self.torque_max_nm, self.torque_max_nm)

    @property
    def detection_can_id(self) -> int:
        return 0x7FF

    @property
    def detection_payload(self) -> list[int]:
        # DM register-read request: read MST_ID (RID 0x07).
        return [self.can_id & 0xFF, (self.can_id >> 8) & 0xFF, 0x33, 0x07]

    def is_detection_reply(
        self,
        response_can_id: int,
        data: Sequence[int],
    ) -> bool:
        return self.resolve_detection_reply(response_can_id, data) is not None

    def resolve_detection_reply(
        self,
        response_can_id: int,
        data: Sequence[int],
        *,
        auto_master_id: bool = False,
    ) -> Optional["GripperMotorProfile"]:
        """Validate a register reply, optionally learning DM MST_ID.

        Zero is a valid explicit MST_ID, never an automatic-mode sentinel.
        No register is written; auto mode requires the uint32 register value
        to agree with the complete, standard 11-bit reply CAN identifier.
        """
        values = [int(value) for value in data]
        if len(values) != 8 or any(value < 0 or value > 0xFF for value in values):
            return None
        response_can_id = int(response_can_id)
        if not 0 <= response_can_id <= 0x7FF:
            return None
        if values[:4] != self.detection_payload:
            return None
        reported_master = struct.unpack("<I", bytes(values[4:8]))[0]
        if reported_master != response_can_id:
            return None
        if not auto_master_id and reported_master != self.master_id:
            return None
        return replace(self, master_id=reported_master)

    def decode_feedback(
        self,
        data: Sequence[int],
        *,
        received_at: float,
    ) -> MotorFeedback:
        return DMMotor.unpack_feedback(
            data,
            received_at=received_at,
            protocol=self.protocol,
            p_range=self.position_range,
            v_range=self.velocity_range,
            t_range=self.torque_range,
        )

    def build_detection_packet(self):
        return DMMotor.build_motor_packet(
            self.bus,
            self.detection_can_id,
            self.detection_payload,
            frame_flags=self.frame_flags,
        )


def select_protocol_candidates(
    mode: GripperProtocolMode | str,
    profiles: Sequence[GripperMotorProfile],
) -> tuple[GripperMotorProfile, ...]:
    """Return the read-only probe candidates allowed for one physical port."""

    try:
        parsed_mode = (
            mode
            if isinstance(mode, GripperProtocolMode)
            else GripperProtocolMode(str(mode).strip().lower())
        )
    except ValueError as exc:
        raise ValueError(
            "DM gripper protocol mode must be 'auto' or 'dm'"
        ) from exc

    candidates = tuple(profiles)
    if not candidates:
        raise ValueError("gripper protocol candidate list must not be empty")
    if len({profile.bus for profile in candidates}) != 1:
        raise ValueError("gripper protocol candidates must use one physical CAN bus")
    if len({profile.label for profile in candidates}) != 1:
        raise ValueError("gripper protocol candidates must describe one gripper side")

    by_protocol: dict[GripperProtocol, GripperMotorProfile] = {}
    for profile in candidates:
        if profile.protocol in by_protocol:
            raise ValueError(
                f"duplicate {profile.protocol.value} gripper protocol candidate"
            )
        by_protocol[profile.protocol] = profile

    if parsed_mode is GripperProtocolMode.AUTO:
        missing = set(GripperProtocol) - set(by_protocol)
        if missing:
            names = ", ".join(sorted(protocol.value for protocol in missing))
            raise ValueError(f"auto gripper protocol mode lacks candidates: {names}")
        return tuple(by_protocol[protocol] for protocol in GripperProtocol)

    protocol = GripperProtocol(parsed_mode.value)
    try:
        return (by_protocol[protocol],)
    except KeyError as exc:
        raise ValueError(
            f"{protocol.value} gripper protocol candidate is unavailable"
        ) from exc


class CalibrationPhase(str, Enum):
    WAITING_FEEDBACK = "waiting_feedback"
    SETTLING = "settling"
    SEEKING_OPEN = "seeking_open"
    RELEASING_INITIAL = "releasing_initial"
    BACKING_OFF_OPEN = "backing_off_open"
    SEEKING_CLOSED = "seeking_closed"
    BACKING_OFF_CLOSED = "backing_off_closed"
    RETURNING_OPEN = "returning_open"
    READY = "ready"
    FAILED = "failed"


@dataclass(frozen=True)
class CalibrationSettings:
    speed_rad_s: float = 0.2
    return_speed_rad_s: float = 0.2
    return_acceleration_rad_s2: float = 1.0
    contact_torque: float = 2.0
    abort_torque: float = 8.0
    contact_confirm_s: float = 0.25
    stopped_velocity_rad_s: float = 0.1
    tracking_error_rad: float = 0.08
    limit_margin_rad: float = 0.15
    minimum_span_rad: float = 1.0
    maximum_search_travel_rad: float = 7.0
    response_timeout_s: float = 1.0
    feedback_timeout_s: float = 0.3
    phase_timeout_s: float = 45.0
    maximum_mos_temperature_c: int = 80
    maximum_motor_temperature_c: int = 80
    maximum_feedback_velocity_rad_s: float = 2.0
    settle_time_s: float = 0.5
    position_min_rad: float = -12.5
    position_max_rad: float = 12.5
    open_direction: int = 1
    minimum_seek_travel_rad: float = 0.1
    stall_timeout_s: float = 3.0
    command_torque_limit_nm: float = 2.5
    kp: float = 5.0
    kd: float = 0.5
    maximum_step_s: float = 0.05
    feedforward_max_nm: float = 1.5
    feedforward_slew_nm_s: float = 1.5
    initial_release_enabled: bool = True
    initial_release_rad: float = 0.2

    def validate(self) -> None:
        positive = (
            self.speed_rad_s, self.return_speed_rad_s, self.return_acceleration_rad_s2,
            self.contact_torque, self.abort_torque,
            self.contact_confirm_s, self.stopped_velocity_rad_s,
            self.tracking_error_rad, self.limit_margin_rad,
            self.minimum_span_rad, self.maximum_search_travel_rad,
            self.response_timeout_s, self.feedback_timeout_s, self.phase_timeout_s,
            self.maximum_mos_temperature_c, self.maximum_motor_temperature_c,
            self.maximum_feedback_velocity_rad_s, self.settle_time_s,
            self.minimum_seek_travel_rad, self.stall_timeout_s,
            self.command_torque_limit_nm, self.kp, self.maximum_step_s,
            self.feedforward_max_nm, self.feedforward_slew_nm_s,
            self.initial_release_rad,
        )
        if not all(math.isfinite(v) and v > 0 for v in positive):
            raise ValueError("gripper calibration timing, gains and limits must be finite and positive")
        if not math.isfinite(self.kd) or not 0 < self.kd <= 5 or self.kp > 500:
            raise ValueError("gripper position-control gains require kp>0 and kd>0 within the MIT encoding range")
        if self.open_direction not in (-1, 1):
            raise ValueError("gripper open_direction must be +1 or -1")
        if not (math.isfinite(self.position_min_rad) and math.isfinite(self.position_max_rad)
                and self.position_min_rad < self.position_max_rad):
            raise ValueError("gripper calibration position range is invalid")
        self._validate_effort_limits()
        if self.feedback_timeout_s > self.response_timeout_s:
            raise ValueError("gripper feedback timeout must not exceed response timeout")
        if self.stall_timeout_s <= self.contact_confirm_s:
            raise ValueError("stall timeout must exceed continuous contact confirmation")
        if self.minimum_seek_travel_rad >= self.maximum_search_travel_rad:
            raise ValueError("minimum seek travel must be smaller than search travel")
        if self.feedforward_max_nm >= self.command_torque_limit_nm:
            raise ValueError("feedforward cap must be below the total command budget")
        if not isinstance(self.initial_release_enabled, bool):
            raise ValueError("initial release enable must be boolean")
        if not self.minimum_seek_travel_rad + 0.02 < self.initial_release_rad < self.minimum_span_rad:
            raise ValueError("initial release must exceed proven travel + tolerance and remain below minimum span")
        if max(self.speed_rad_s, self.return_speed_rad_s) >= self.maximum_feedback_velocity_rad_s:
            raise ValueError("calibration speeds must be below calibration feedback speed guard")


    def _validate_effort_limits(self):
        if not self.contact_torque < self.command_torque_limit_nm < self.abort_torque:
            raise ValueError("require contact torque < command torque budget < abort torque")


def bounded_position_target(target: float, feedback: MotorFeedback, *,
                            kp: float, kd: float, torque_limit: float,
                            velocity_desired: float = 0.0) -> float:
    """Bound the *estimated* MIT PD effort; not a hardware torque limiter."""
    if not all(math.isfinite(v) for v in
               (target, feedback.position, feedback.velocity, kp, kd, torque_limit, velocity_desired)):
        raise ValueError("non-finite MIT target/feedback/gain")
    if not 0 < kp <= 500 or not 0 <= kd <= 5 or torque_limit <= 0:
        raise ValueError("invalid MIT PD gains or torque budget")
    # PD budget includes the commanded MIT velocity, not just position/FF.
    damping = kd * (feedback.velocity - velocity_desired)
    lower = feedback.position + (damping - torque_limit) / kp
    upper = feedback.position + (damping + torque_limit) / kp
    return max(lower, min(float(target), upper))


class FrictionAssist:
    """Bounded MIT t_ff assistance, not a hardware current/speed limiter.

    Build effort only while fresh feedback lags the position trajectory. Stop
    building at the requested speed; remove assistance on arrival/reversal or
    pause. The position target never moves outside its geometric limits.
    """

    def __init__(self, maximum_nm: float, slew_nm_s: float) -> None:
        if not all(math.isfinite(v) and v > 0 for v in (maximum_nm, slew_nm_s)):
            raise ValueError("invalid feedforward cap/slew")
        self.maximum_nm, self.slew_nm_s = maximum_nm, slew_nm_s
        self.reset()

    def reset(self) -> None:
        self.value = 0.0
        self._feedback_at = None

    def update(self, target: float, feedback: MotorFeedback, dt: float, *,
               kp: float, kd: float, budget: float, speed: float,
               active: bool = True, tolerance: float = 0.005,
               velocity_desired: float = 0.0) -> float:
        if not all(math.isfinite(v) for v in (target, feedback.position,
                feedback.velocity, feedback.received_at, dt, kp, kd, budget,
                speed, tolerance, velocity_desired)) or dt < 0 or min(budget, speed, tolerance) <= 0 \
                or not 0 < kp <= 500 or not 0 < kd <= 5:
            raise ValueError("invalid friction assistance input")
        position_error = target - feedback.position
        # Position can be projected behind feedback to budget a positive v_des.
        # Friction compensation follows the combined command, not that projected
        # position alone (which would incorrectly brake the intended motion).
        error = position_error + kd * velocity_desired / kp
        if not active or abs(error) <= tolerance:
            self.reset()
            self._feedback_at = feedback.received_at
            return 0.0
        if self._feedback_at is not None:
            elapsed = feedback.received_at - self._feedback_at
            if elapsed < 0:
                raise ValueError("feedforward feedback clock moved backwards")
            dt = min(dt, elapsed)
        self._feedback_at = feedback.received_at
        direction = math.copysign(1.0, error)
        if self.value * direction < 0:
            self.value = 0.0  # never carry old-direction effort through reversal
        change = self.slew_nm_s * min(dt, 0.05)
        along_speed = direction * feedback.velocity
        if along_speed >= speed:
            magnitude = max(0.0, abs(self.value) - 4.0 * change)
        else:
            magnitude = abs(self.value) + change * max(0.0, min(1.0, 1.0 - along_speed / speed))
        pd = kp * position_error + kd * (velocity_desired - feedback.velocity)
        # Sum of PD + t_ff, not two independent allowances. Leave a small
        # quantization/feedback headroom where PD has not consumed the budget.
        available = max(0.0, budget * 0.95 - direction * pd)
        self.value = direction * min(magnitude, self.maximum_nm, available)
        return self.value


class GripperCalibrator:
    """Feedback-clocked homing. Ambiguous stalls fail; they never create limits.

    Opening direction is per motor, not inferred from its power-up position.
    Each accepted endpoint requires real travel in the search direction.
    An initial suspected open stop is never accepted without travel. A single
    bounded inward release and repeatable re-seek may establish it; failure to
    release, insufficient travel or a displaced contact always fails closed.
    """

    _TORQUE_FILTER_ALPHA = 0.2

    def __init__(self, label: str, settings: CalibrationSettings) -> None:
        settings.validate()
        self.label = str(label)
        self.settings = settings
        self.reset(0.0)

    @property
    def ready(self) -> bool:
        return self.phase is CalibrationPhase.READY

    @property
    def failed(self) -> bool:
        return self.phase is CalibrationPhase.FAILED

    def reset(self, now: float) -> None:
        if not math.isfinite(now):
            raise ValueError("calibration reset time must be finite")
        self.phase = CalibrationPhase.WAITING_FEEDBACK
        self.target_position: Optional[float] = None
        self.open_limit: Optional[float] = None
        self.closed_limit: Optional[float] = None
        self.open_position: Optional[float] = None
        self.closed_position: Optional[float] = None
        self.failure_reason: Optional[str] = None
        self._started_at = float(now)
        self._phase_started_at = float(now)
        self._search_started_position = 0.0
        self._filtered_torque: Optional[float] = None
        self._torque_bias = 0.0
        self._contact_since: Optional[float] = None
        self._stalled_since: Optional[float] = None
        self._settled_since: Optional[float] = None
        self._last_feedback_position: Optional[float] = None
        self._last_feedback_at: Optional[float] = None
        self._last_update_at: Optional[float] = None
        self._last_feedback: Optional[MotorFeedback] = None
        self.last_contact_reason: Optional[str] = None
        self._assist = FrictionAssist(self.settings.feedforward_max_nm,
                                      self.settings.feedforward_slew_nm_s)
        self._progress_position: Optional[float] = None
        self._initial_contact_position: Optional[float] = None
        self._initial_release_goal: Optional[float] = None
        self._terminal_hold = False
        self._terminal_goal: Optional[float] = None
        self._return_velocity = 0.0

    @property
    def feedforward_torque(self) -> float:
        return self._assist.value

    def diagnostics(self) -> dict:
        f = self._last_feedback
        return dict(
            phase=self.phase.value,
            position=None if f is None else f.position,
            target=self.target_position,
            velocity=None if f is None else f.velocity,
            torque=None if f is None else f.torque,
            filtered_torque=self._filtered_torque, torque_bias=self._torque_bias,
            actual_travel=None if f is None else f.position - self._search_started_position,
            open_direction=self.settings.open_direction,
            contact_reason=self.last_contact_reason, failure=self.failure_reason,
            t_ff=self.feedforward_torque,
            terminal_hold=self._terminal_hold,
            terminal_goal=self._terminal_goal,
            terminal_error=(None if f is None or self._terminal_goal is None else
                            self._terminal_goal - f.position),
            settle_tolerance=self._settle_tolerance(),
            settled_for_s=(0.0 if self._settled_since is None or self._last_update_at is None
                           else max(0.0, self._last_update_at - self._settled_since)),
            estimated_total_torque=(None if f is None or self.target_position is None else
                self.settings.kp * (self.target_position - f.position)
                - self.settings.kd * f.velocity + self.feedforward_torque),
        )

    def fail(self, reason: str) -> None:
        if not self.failed:
            self.failure_reason = str(reason)
            self.phase = CalibrationPhase.FAILED
            self._assist.reset()
            self._terminal_hold = False

    def _fail_with_sample(self, reason: str) -> None:
        # Preserve the pre-failure phase, target and measured evidence in logs.
        detail = self.diagnostics()
        self.fail(f"{self.label} gripper {reason}; sample={detail}")

    def _set_phase(self, phase: CalibrationPhase, now: float,
                   feedback: MotorFeedback) -> None:
        self.phase = phase
        self._phase_started_at = now
        self._search_started_position = feedback.position
        self._contact_since = self._stalled_since = self._settled_since = None
        self._progress_position = feedback.position
        self._assist.reset()
        self._terminal_hold = False
        self._terminal_goal = None
        self._return_velocity = 0.0
        # Do NOT relearn bias at a loaded hard stop on a phase transition.

    def _contact_confirmed(self, feedback: MotorFeedback, now: float) -> bool:
        assert self.target_position is not None and self._filtered_torque is not None
        direction = self.settings.open_direction * (
            1 if self.phase is CalibrationPhase.SEEKING_OPEN else -1)
        error = direction * (self.target_position - feedback.position)
        candidate = (abs(feedback.velocity) <= self.settings.stopped_velocity_rad_s
                     and error >= self.settings.tracking_error_rad)
        torque_contact = abs(self._filtered_torque - self._torque_bias) >= self.settings.contact_torque
        if candidate:
            if self._stalled_since is None:
                self._stalled_since = now
        else:
            self._stalled_since = None
        # Require a continuous conjunction of DM endpoint conditions.
        if candidate and torque_contact:
            if self._contact_since is None:
                self._contact_since = now
            if now - self._contact_since >= self.settings.contact_confirm_s:
                travel = direction * (feedback.position - self._search_started_position)
                if travel < self.settings.minimum_seek_travel_rad:
                    if (self.phase is CalibrationPhase.SEEKING_OPEN
                            and self.settings.initial_release_enabled
                            and self._initial_contact_position is None):
                        goal = feedback.position - direction * self.settings.initial_release_rad
                        if not self.settings.position_min_rad < goal < self.settings.position_max_rad:
                            self._fail_with_sample("initial release goal outside protocol range")
                            return False
                        self._initial_contact_position = feedback.position
                        self._initial_release_goal = goal
                        self._set_phase(CalibrationPhase.RELEASING_INITIAL, now, feedback)
                        self.target_position = feedback.position  # unload before reversing
                        return False
                    self._fail_with_sample("contact without proven search travel; initial endpoint/obstruction is ambiguous")
                    return False
                if (self.phase is CalibrationPhase.SEEKING_OPEN
                        and self._initial_contact_position is not None
                        and abs(feedback.position - self._initial_contact_position) > self.settings.tracking_error_rad):
                    self._fail_with_sample("initial contact was not repeatable after release")
                    return False
                self.last_contact_reason = "continuous_torque_speed_error_and_actual_travel"
                return True
        else:
            self._contact_since = None
        if self._stalled_since is not None and now - self._stalled_since >= self.settings.stall_timeout_s:
            self._fail_with_sample("unconfirmed stall; no valid mechanical endpoint established")
        return False

    def _settle_tolerance(self) -> float:
        return min(self.settings.tracking_error_rad * 0.5,
                   self.settings.limit_margin_rad * 0.25, 0.01)

    def _update_goal_assist(self, goal: float, feedback: MotorFeedback, dt: float) -> None:
        """Unload inside a terminal band, then prove a fresh, unassisted hold.

        The old independent .005-rad FF deadband restarted assistance inside
        the .01-rad settling tolerance, resetting dwell at each encoder toggle.
        Latch unloading at 75% of the SAME acceptance tolerance; keep it until
        error leaves the full band. Do not change the goal, increase tolerance,
        or suppress feedback/velocity/torque checks while holding.
        """
        tolerance = self._settle_tolerance()
        at_goal = self.target_position is not None and abs(self.target_position - goal) < 1e-8
        error = abs(feedback.position - goal)
        if self._terminal_goal != goal or not at_goal or error > tolerance:
            self._terminal_hold = False
            self._settled_since = None
        self._terminal_goal = goal
        if at_goal and error <= tolerance * 0.75:
            self._terminal_hold = True
        if self._terminal_hold:
            # Keep the fixed-position PD hold but remove friction assistance.
            # A subsequent .5s fresh-feedback dwell proves arrival without FF.
            self._assist.reset()
        else:
            self._assist.update(self.target_position, feedback, dt,
                kp=self.settings.kp, kd=self.settings.kd,
                budget=self.settings.command_torque_limit_nm,
                speed=(self.settings.return_speed_rad_s if self.phase is CalibrationPhase.RETURNING_OPEN
                       else self.settings.speed_rad_s), tolerance=min(0.005, tolerance * 0.5))

    def _stable_at(self, goal: float, feedback: MotorFeedback, now: float) -> bool:
        tolerance = self._settle_tolerance()
        valid = (
            self.target_position is not None and abs(self.target_position - goal) < 1e-8
            and (self.phase is CalibrationPhase.SETTLING or self._terminal_hold)
            and abs(feedback.position - goal) <= tolerance
            and abs(feedback.velocity) <= self.settings.stopped_velocity_rad_s
            and self._filtered_torque is not None
            and abs(self._filtered_torque - self._torque_bias) < self.settings.contact_torque
            and self.feedforward_torque == 0.0
        )
        if not valid:
            self._settled_since = None
            return False
        if self._settled_since is None:
            self._settled_since = now
        return now - self._settled_since >= self.settings.settle_time_s

    def _movement_stalled(self, feedback: MotorFeedback, now: float) -> bool:
        assert self.target_position is not None
        # Measured progress, not just instantaneous velocity (which can be
        # noisy near zero). Small terminal errors must not wait 50 seconds.
        error = self.target_position - feedback.position
        progress = (0.0 if self._progress_position is None else
                    math.copysign(1.0, error) * (feedback.position - self._progress_position))
        if abs(error) > self._settle_tolerance():
            if progress >= 0.005:
                self._progress_position = feedback.position
                self._stalled_since = now
            if self._stalled_since is None:
                self._stalled_since = now
                self._progress_position = feedback.position
            elif now - self._stalled_since >= self.settings.stall_timeout_s:
                self._fail_with_sample("backoff/return stalled; stopping without waiting for phase timeout")
        else:
            self._stalled_since = None
            self._progress_position = feedback.position
        return self.failed

    def _check_common_faults(self, feedback: Optional[MotorFeedback], now: float) -> bool:
        if feedback is None:
            if now - self._started_at >= self.settings.response_timeout_s:
                self._fail_with_sample("motor is offline: no response frame")
            return self.failed
        values = (feedback.position, feedback.velocity, feedback.torque,
                  feedback.mos_temperature_c, feedback.motor_temperature_c, feedback.received_at)
        if not all(math.isfinite(v) for v in values):
            self._fail_with_sample("non-finite feedback")
        elif feedback.received_at > now + 0.05:
            self._fail_with_sample("feedback timestamp is in the future")
        elif now - feedback.received_at >= self.settings.feedback_timeout_s:
            self._fail_with_sample("motor is offline: feedback timed out")
        elif feedback.status is not None and feedback.status != 1:
            self._fail_with_sample(f"DM motor status 0x{feedback.status:X}")
        elif feedback.mos_temperature_c >= self.settings.maximum_mos_temperature_c:
            self._fail_with_sample("MOS temperature is too high")
        elif feedback.motor_temperature_c >= self.settings.maximum_motor_temperature_c:
            self._fail_with_sample("motor temperature is too high")
        elif abs(feedback.velocity) >= self.settings.maximum_feedback_velocity_rad_s:
            self._fail_with_sample("feedback velocity is implausible")
        elif abs(feedback.torque) >= self.settings.abort_torque:
            # Raw absolute torque: filtering/bias must never hide an overload.
            self._fail_with_sample("raw torque exceeded the abort limit")
        elif not self.settings.position_min_rad < feedback.position < self.settings.position_max_rad:
            self._fail_with_sample("feedback reached the protocol position limit")
        return self.failed

    def _bounded_target(self, desired: float, feedback: MotorFeedback) -> float:
        return bounded_position_target(desired, feedback, kp=self.settings.kp,
                                       kd=self.settings.kd,
                                       torque_limit=self.settings.command_torque_limit_nm)

    def _move_toward(self, goal: float, step: float, feedback: MotorFeedback) -> None:
        assert self.target_position is not None
        delta = max(-step, min(step, goal - self.target_position))
        self.target_position = self._bounded_target(self.target_position + delta, feedback)

    def update(self, feedback: Optional[MotorFeedback], now: float,
               dt: float) -> Optional[float]:
        if self.failed:
            return None
        if not all(math.isfinite(v) for v in (now, dt)) or dt < 0:
            self._fail_with_sample("invalid control time/dt")
            return None
        if self._last_update_at is not None and now < self._last_update_at:
            self._fail_with_sample("control clock moved backwards")
            return None
        self._last_update_at = now
        if feedback is not None:
            self._last_feedback = feedback
        if self._check_common_faults(feedback, now):
            return None
        if feedback is None:
            return None
        if self._last_feedback_at is not None:
            elapsed = feedback.received_at - self._last_feedback_at
            if elapsed < 0:
                self._fail_with_sample("feedback clock moved backwards")
                return None
            if elapsed == 0:
                # A cached sample cannot advance target, filters or contact dwell.
                return self.target_position
            # Advance on the feedback clock, not one (possibly faster) body tick.
            dt = min(elapsed, self.settings.maximum_step_s)
            assert self._last_feedback_position is not None
            if abs(feedback.position - self._last_feedback_position) > self._feedback_position_step_limit(feedback, elapsed):
                self._fail_with_sample("feedback position jumped unexpectedly")
                return None
        self._last_feedback_at = feedback.received_at
        self._last_feedback_position = feedback.position
        if self._filtered_torque is None:
            self._filtered_torque = feedback.torque
        else:
            self._filtered_torque += self._TORQUE_FILTER_ALPHA * (feedback.torque - self._filtered_torque)
        if self.ready:
            assert self.target_position is not None
            if abs(feedback.position - self.target_position) > self.settings.tracking_error_rad:
                self._fail_with_sample("calibrated holding position drifted while waiting for partner")
                return None
            estimated = self.settings.kp * (self.target_position - feedback.position) - self.settings.kd * feedback.velocity
            if abs(estimated) > self.settings.command_torque_limit_nm:
                self._fail_with_sample("calibrated holding command exceeds PD effort budget")
                return None
            return self.target_position
        if self.phase is CalibrationPhase.WAITING_FEEDBACK:
            self.target_position = feedback.position
            self._set_phase(CalibrationPhase.SETTLING, now, feedback)
            return self.target_position
        assert self.target_position is not None
        if now - self._phase_started_at >= self.settings.phase_timeout_s:
            self._fail_with_sample("calibration phase timed out")
            return None
        step = self.settings.speed_rad_s * min(dt, self.settings.maximum_step_s)
        if self.phase is CalibrationPhase.SETTLING:
            if self._stable_at(self.target_position, feedback, now):
                # Only a small unloaded offset is accepted; never adapt at contact.
                # Leave headroom for the contact threshold and filter dwell
                # below the command budget, even with a nonzero idle offset.
                bias_limit = min(self.settings.contact_torque * 0.25,
                                 (self.settings.command_torque_limit_nm - self.settings.contact_torque) * 0.25)
                self._torque_bias = max(-bias_limit, min(self._filtered_torque, bias_limit))
                self._set_phase(CalibrationPhase.SEEKING_OPEN, now, feedback)
            elif now - self._phase_started_at >= self.settings.stall_timeout_s:
                self._fail_with_sample("initial hold failed to settle; refusing to start search")
            return self.target_position

        if self.phase is CalibrationPhase.RELEASING_INITIAL:
            assert self._initial_release_goal is not None and self._initial_contact_position is not None
            inward = -self.settings.open_direction
            travel = inward * (feedback.position - self._initial_contact_position)
            if not -0.02 <= travel <= self.settings.initial_release_rad + 0.02:
                self._fail_with_sample("initial release moved outside bounded travel")
                return None
            self._move_toward(self._initial_release_goal, step, feedback)
            self._update_goal_assist(self._initial_release_goal, feedback, dt)
            if self._movement_stalled(feedback, now):
                return None
            if self._stable_at(self._initial_release_goal, feedback, now):
                if travel < self.settings.minimum_seek_travel_rad + 0.02:
                    self._fail_with_sample("initial release did not prove sufficient movement")
                    return None
                self.target_position = feedback.position
                self._set_phase(CalibrationPhase.SEEKING_OPEN, now, feedback)
            return self.target_position

        if self.phase in (CalibrationPhase.SEEKING_OPEN, CalibrationPhase.SEEKING_CLOSED):
            opening = self.phase is CalibrationPhase.SEEKING_OPEN
            direction = self.settings.open_direction * (1 if opening else -1)
            self.target_position = self._bounded_target(self.target_position + direction * step, feedback)
            travel = direction * (feedback.position - self._search_started_position)
            if (opening and self._initial_contact_position is not None
                    and direction * (feedback.position - self._initial_contact_position) > self.settings.tracking_error_rad):
                self._fail_with_sample("initial re-seek passed the suspected endpoint; refusing extended search")
            elif travel < -self.settings.tracking_error_rad:
                self._fail_with_sample("feedback moved opposite to commanded search direction")
            elif max(abs(feedback.position - self._search_started_position),
                     abs(self.target_position - self._search_started_position)) >= self.settings.maximum_search_travel_rad:
                self._fail_with_sample("actual/target exceeded maximum calibration travel")
            elif not self.settings.position_min_rad < self.target_position < self.settings.position_max_rad:
                self._fail_with_sample("target reached the protocol position limit")
            elif self._contact_confirmed(feedback, now):
                if opening:
                    self.open_limit = feedback.position
                    self.open_position = self.open_limit - direction * self.settings.limit_margin_rad
                    self._set_phase(CalibrationPhase.BACKING_OFF_OPEN, now, feedback)
                else:
                    self.closed_limit = feedback.position
                    self.closed_position = self.closed_limit - direction * self.settings.limit_margin_rad
                    assert self.open_position is not None
                    span = self.settings.open_direction * (self.open_position - self.closed_position)
                    if span < self.settings.minimum_span_rad:
                        self._fail_with_sample(f"calibrated span is too small: {span:.4f} rad")
                        return None
                    self._set_phase(CalibrationPhase.BACKING_OFF_CLOSED, now, feedback)
                # Immediately unload the accumulated forward error. This may be
                # a large target reduction, but never a jump *into* the stop.
                self.target_position = feedback.position
            return None if self.failed else self.target_position

        if self.phase in (CalibrationPhase.BACKING_OFF_OPEN,
                          CalibrationPhase.BACKING_OFF_CLOSED, CalibrationPhase.RETURNING_OPEN):
            phase = self.phase
            goal = self.closed_position if phase is CalibrationPhase.BACKING_OFF_CLOSED else self.open_position
            assert goal is not None
            if not self.settings.position_min_rad < goal < self.settings.position_max_rad:
                self._fail_with_sample("backoff goal outside protocol range")
                return None
            if phase is CalibrationPhase.RETURNING_OPEN:
                proposed, self._return_velocity = position_trajectory_step(
                    self.target_position, self._return_velocity, goal,
                    min(dt, self.settings.maximum_step_s),
                    self.settings.return_speed_rad_s, self.settings.return_acceleration_rad_s2)
                self.target_position = self._bounded_target(proposed, feedback)
                if abs(self.target_position - proposed) > 1e-9:
                    self._return_velocity = 0.0  # no trajectory windup under effort saturation
            else:
                self._move_toward(goal, step, feedback)
            # Fixed safe target, independent bounded t_ff for static friction.
            self._update_goal_assist(goal, feedback, dt)
            direction = self.settings.open_direction
            if ((self.open_limit is not None and
                    direction * (feedback.position - self.open_limit) > 0.02) or
                    (self.closed_limit is not None and
                    direction * (self.closed_limit - feedback.position) > 0.02)):
                self._fail_with_sample("return feedback outside measured hard limits")
                return None
            if self._movement_stalled(feedback, now):
                return None
            if self._stable_at(goal, feedback, now):
                if phase is CalibrationPhase.BACKING_OFF_OPEN:
                    self._set_phase(CalibrationPhase.SEEKING_CLOSED, now, feedback)
                elif phase is CalibrationPhase.BACKING_OFF_CLOSED:
                    self._set_phase(CalibrationPhase.RETURNING_OPEN, now, feedback)
                else:
                    self._set_phase(CalibrationPhase.READY, now, feedback)
            return self.target_position
        self._fail_with_sample("invalid calibration phase")
        return None

    def _feedback_position_step_limit(self, feedback, elapsed):
        return max(0.1, abs(feedback.velocity) * elapsed * 2.0 + 0.02)


def position_trajectory_step(position: float, velocity: float, goal: float,
                             dt: float, speed: float, acceleration: float,
                             braking_acceleration: Optional[float] = None) -> tuple[float, float]:
    """Exact constant-acceleration segments, replanned for the latest position goal.

    A reversed/nearer goal first brakes existing momentum, rather than snapping
    velocity to zero at a crossed target. Caller retains geometry/effort guards.
    This is a reference trajectory, NOT a guarantee of actual braking distance.
    """
    brake = acceleration if braking_acceleration is None else braking_acceleration
    if not all(math.isfinite(x) for x in (position, velocity, goal, dt, speed, acceleration, brake)) \
            or dt < 0 or min(speed, acceleration, brake) <= 0:
        raise ValueError("invalid position trajectory input")
    p, v, remaining = position, velocity, dt
    for _ in range(8):
        if remaining <= 1e-12:
            break
        delta = goal - p
        if abs(delta) < 1e-12 and abs(v) < 1e-12:
            return goal, 0.0
        direction = math.copysign(1.0, delta)
        u, distance = direction * v, abs(delta)
        if u < -1e-12 or (u > 1e-12 and u*u/(2*brake) >= distance - 1e-12):
            a = -math.copysign(brake, v)
            duration = abs(v) / brake
        elif u > speed + 1e-12:
            a, duration = -direction*brake, (u-speed)/brake
        else:
            peak = min(speed, math.sqrt(max(0.0, (2*acceleration*brake*distance + brake*u*u)/(acceleration+brake))))
            if peak > u + 1e-10:
                a, duration = direction*acceleration, (peak-u)/acceleration
            else:
                a = 0.0
                duration = max(0.0, distance-u*u/(2*brake)) / max(u, 1e-12)
                if duration < 1e-12:
                    a, duration = -direction*brake, max(u, 0.0)/brake
        step = min(remaining, duration)
        if step <= 1e-12:
            break
        p += v*step + .5*a*step*step
        v += a*step
        if abs(v) < 1e-10:
            v = 0.0
        remaining -= step
    if abs(p-goal) < 1e-10 and abs(v) < 1e-9:
        return goal, 0.0
    return p, v


@dataclass(frozen=True)
class GripMotionSettings:
    command_nm: float = 2.5
    hold_nm: float = 2.0
    contact_nm: float = 1.8
    soft_nm: float = 3.0
    hard_nm: float = 4.0
    acceleration: float = 1.0
    braking_acceleration: float = 1.0
    confirm_s: float = 0.25
    stall_s: float = 1.5
    tracking_error: float = 0.08
    stopped_velocity: float = 0.1
    kp: float = 10.0
    kd: float = 0.5
    ff_max: float = 2.2
    ff_slew: float = 1.5
    velocity_feedforward: bool = False

    def validate(self):
        if not isinstance(self.velocity_feedforward, bool):
            raise ValueError("velocity feedforward must be boolean")
        if not all(math.isfinite(v) and v > 0 for k, v in vars(self).items() if k != 'velocity_feedforward'):
            raise ValueError("grip motion settings must be finite and positive")
        self._validate_effort_limits()
        if self.confirm_s >= self.stall_s or self.kp > 500 or self.kd > 5:
            raise ValueError("invalid grip confirmation/stall time or MIT gains")

    def _validate_effort_limits(self):
        if not self.contact_nm <= self.hold_nm <= self.command_nm < self.soft_nm < self.hard_nm:
            raise ValueError("require contact <= hold <= command < soft < hard torque limits")


class MotionResponseTiming:
    """Feedback-only diagnostic. Never treats a completed reference as arrival."""
    def __init__(self):
        self.pending = None
        self._last_at = None

    def observe(self, feedback, now, desired, active, mode, received_at=None):
        if self._last_at is not None and feedback.received_at <= self._last_at:
            return None
        self._last_at = feedback.received_at
        if not active:
            self.pending = None
            return None
        if self.pending is None:
            if abs(desired-feedback.position) <= .02 or mode != 'tracking':
                return None
            self.pending = dict(started_at=now if received_at is None else min(now,received_at),
                start_position=feedback.position, goal=desired, changes=0, arrived_at=None)
        p = self.pending
        if abs(desired-p['goal']) > .005:
            p['goal'], p['arrived_at'] = desired, None
            p['changes'] += 1
        if mode != 'tracking':
            self.pending = None
            return dict(outcome=mode, elapsed_s=now-p['started_at'], start_position=p['start_position'],
                        goal=p['goal'], actual_position=feedback.position, input_changes=p['changes'])
        if abs(feedback.position-desired) <= .02 and abs(feedback.velocity) <= .1:
            if p['arrived_at'] is None:
                p['arrived_at'] = now
            if now-p['arrived_at'] >= .05:
                self.pending = None
                return dict(outcome='arrived', elapsed_s=now-p['started_at'],
                    first_arrival_s=p['arrived_at']-p['started_at'],
                    start_position=p['start_position'],goal=desired,actual_position=feedback.position,
                    input_changes=p['changes'], tolerance_rad=.02, stable_s=.05)
        else:
            p['arrived_at'] = None
        return None


class GripMotionController:
    """Bounded position compliance, not a certified force/current controller.

    Normal resistance never creates new limits or counts as a successful
    close. Contact holds a FIXED virtual spring anchor without ratcheting the
    target farther into an object. Opening or uncertain stalls latch locally.
    The caller must still check raw hard torque, freshness, status, temperature
    and geometric feedback limits before this controller can issue commands.
    """
    def __init__(self, settings: GripMotionSettings, target: float):
        settings.validate()
        if not math.isfinite(target):
            raise ValueError("invalid grip target")
        self.settings = settings
        self.target = float(target)
        self._effective_settings = settings
        self.effective_speed = 0.0
        self._rate_scale = 1.0
        self._speed_scale = 1.0
        self._gain_scale = 1.0
        self._feedback_periods = deque(maxlen=21)
        self.rate_reason = 'nominal'
        self.command_position = float(target)
        self.command_kp = settings.kp
        self.command_kd = settings.kd
        self.command_feedforward = 0.0
        self.velocity = 0.0  # trajectory velocity, not measured motor velocity
        self.command_velocity = 0.0
        self.assist = FrictionAssist(settings.ff_max, settings.ff_slew)
        self.mode = "tracking"
        self.reason = None
        self.anchor = None
        self.blocked_desired = None
        self.blocked_direction = 0.
        self._last_at = None
        self._last_now = None
        self._filtered_torque = None
        self._contact_since = self._stall_since = None
        self._terminal_hold = False
        self.response_timing = MotionResponseTiming()

    @property
    def locally_held(self):
        return self.mode in ("contact_hold", "blocked_hold")

    def pause(self):
        self.velocity = 0.
        self.command_velocity = 0.
        if not self.locally_held:
            self.command_feedforward = 0.
        self.assist.reset()
        self._contact_since = self._stall_since = None

    def block(self, feedback, desired, reason):
        self.mode, self.reason = "blocked_hold", reason
        self.anchor = feedback.position
        self.blocked_desired = desired
        self.blocked_direction = math.copysign(1., desired-feedback.position)
        self.pause()

    def _bounded(self, target, f, lower, upper, budget):
        target = max(lower, min(upper, target))
        if self._effective_settings.velocity_feedforward:
            # Joint feasibility of position, velocity and effort. Near a soft
            # endpoint a saturated velocity term must be reduced, not force
            # the position command outside the calibrated interval.
            kd, kp = self._effective_settings.kd, self._effective_settings.kp
            vmin = f.velocity + (-budget-kp*(upper-f.position))/kd
            vmax = f.velocity + (budget-kp*(lower-f.position))/kd
            self.command_velocity = max(vmin, min(self.command_velocity, vmax))
        target = bounded_position_target(target, f, kp=self._effective_settings.kp,
                                         kd=self._effective_settings.kd, torque_limit=budget,
                                         velocity_desired=self.command_velocity)
        if not lower-1e-9 <= target <= upper+1e-9:
            raise ValueError("soft limits and grip command effort have no feasible intersection")
        return max(lower, min(upper, target))

    def _wire_command(self, f, lower, upper, ff, budget):
        """Keep high-rate tracking damping OUT of the motor's impact response.

        A motor-local kd=3 and v_des=4 can add ~12 Nm when an object abruptly
        stops a moving jaw, even though the last sampled total was only 2.5.
        Compute the bounded tracking effort above, then realize it with a low
        local slope. This remains MIT position control with finite kp/kd; no
        current-mode/register change. Delayed/lost feedback still faults.
        """
        s = self._effective_settings
        effort = s.kp*(self.target-f.position) + s.kd*(self.command_velocity-f.velocity) + ff
        if abs(effort) > budget+1e-8:
            raise ValueError("tracking effort exceeded combined budget")
        if s.velocity_feedforward:
            self.command_kp, self.command_kd = 5.0, .05
            self.command_velocity = max(-self.effective_speed, min(self.command_velocity, self.effective_speed))
            if self._terminal_hold and self.mode == 'tracking':
                # Stable arrival unloads active tracking compensation; a small
                # passive position spring remains, exactly as in prior homing.
                self.command_velocity = 0.0
                self.command_position = self.target
                self.command_feedforward = 0.0
                return self.command_position, 0.0
            wire = f.position + (effort-self.command_kd*(self.command_velocity-f.velocity)-ff)/self.command_kp
            wire = max(lower, min(upper, wire))
            pd = self.command_kp*(wire-f.position)+self.command_kd*(self.command_velocity-f.velocity)
            # If the soft limit prevents representing all bounded effort as a
            # position spring, use remaining FF allowance (never increase cap).
            ff = max(-s.ff_max, min(effort-pd, s.ff_max))
            ff = max(-budget-pd, min(ff, budget-pd))
            if abs(ff) > s.ff_max+1e-8:
                raise ValueError("wire geometry/effort/FF constraints incompatible")
            self.command_position = wire
        else:
            self.command_position = self.target
            self.command_kp, self.command_kd = s.kp, s.kd
        self.command_feedforward = ff
        return self.command_position, ff

    def update(self, f: MotorFeedback, now: float, dt: float, *, desired: float,
               open_position: float, closed_position: float, speed: float,
               active: bool, fast_follow: bool = True) -> tuple[float, float]:
        s = self.settings
        if not all(math.isfinite(x) for x in (now, dt, desired, open_position,
                closed_position, speed, f.position, f.velocity, f.torque, f.received_at)) \
                or dt < 0 or speed <= 0 or open_position == closed_position:
            raise ValueError("invalid grip motion input")
        if self._last_now is not None and now < self._last_now:
            raise ValueError("grip control clock moved backwards")
        self._last_now = now
        if self._last_at is not None and f.received_at < self._last_at:
            raise ValueError("grip feedback clock moved backwards")
        fresh = self._last_at is None or f.received_at > self._last_at
        step_dt = min(.05, dt if self._last_at is None else max(0., f.received_at-self._last_at))
        if not active:
            self.pause()
        if not fresh:
            if fast_follow and s.velocity_feedforward and now-f.received_at > .05:
                # Do not keep driving a fast velocity/FF command on old data.
                # The owning state retains its independent hard timeout/fault.
                self.pause()
                self.target = self.command_position = max(min(open_position, closed_position),
                    min(max(open_position, closed_position), f.position))
                self.command_kp, self.command_kd = 5.0, .05
                self.command_feedforward = 0.0
                self.rate_reason = 'feedback_age_hold'
            return self.command_position, self.command_feedforward
        if s.velocity_feedforward:
            if fast_follow:
                if self._last_at is not None:
                    self._feedback_periods.append(max(.001, f.received_at-self._last_at))
                periods = sorted(self._feedback_periods)
                period = periods[int(.9*(len(periods)-1))] if periods else .01
                # Separate stable loop bandwidth from trajectory speed: normal
                # 40-50 Hz feedback is NOT a request to halve trajectory speed.
                gain_goal = min(1.0, .01/max(.001, period, step_dt))
                self._gain_scale = min(gain_goal, self._gain_scale + step_dt)
                gain_scale = self._gain_scale
                speed_goal = min(1.0, .01/period) if period > .025 else 1.0
                if step_dt > .035:
                    speed_goal = min(speed_goal, .01/step_dt)
                self._speed_scale = min(speed_goal, self._speed_scale + 2.0*step_dt)
                self.rate_reason = ('feedback_gap' if step_dt > .035 else
                                    'low_feedback_rate' if period > .025 else
                                    'recovering' if self._speed_scale < .999 else 'nominal')
                s = replace(s, kp=s.kp*gain_scale, kd=s.kd*gain_scale,
                            acceleration=s.acceleration*self._speed_scale,
                            # Start deceleration earlier at lower control
                            # bandwidth, without slowing the entire stroke.
                            braking_acceleration=s.braking_acceleration*max(.5, gain_scale))
                speed *= self._speed_scale
            else:
                # Preserve the previously accepted normal-exit profile.
                observed_scale = (1.0 if self._last_at is None else min(1.0, .01/max(step_dt, .001)))
                if not self.locally_held and abs(f.velocity) <= s.stopped_velocity and (not active or self._terminal_hold):
                    self._rate_scale = observed_scale
                else:
                    self._rate_scale = min(self._rate_scale, observed_scale)
                scale = self._rate_scale
                s = replace(s, kp=s.kp*scale, kd=s.kd*scale,
                            acceleration=s.acceleration*scale,
                            braking_acceleration=s.braking_acceleration*scale)
                speed *= scale
                self.rate_reason = 'legacy_exit_profile'
        self._effective_settings = s
        self.effective_speed = speed
        self._last_at = f.received_at
        lower, upper = sorted((open_position, closed_position))
        desired = max(lower, min(upper, desired))
        opening = math.copysign(1., open_position-closed_position)
        closing = -opening
        self._filtered_torque = (f.torque if self._filtered_torque is None else
                                self._filtered_torque + .2*(f.torque-self._filtered_torque))

        if abs(f.torque) >= s.soft_nm:
            if self.mode != "blocked_hold":
                self.block(f, desired, "feedback_soft_overload")
        elif self.locally_held and active:
            release = (opening*(desired-f.position) > .02 if self.mode == "contact_hold" else
                       self.blocked_direction*(desired-self.blocked_desired) < -.03)
            if release:
                self.mode, self.reason, self.anchor = "tracking", None, None
                self.target = max(lower, min(upper, f.position))
                self._terminal_hold = False
                self.pause()

        if not self.locally_held and active:
            # Velocity feedforward also contributes contact effort. Checking
            # position error alone would miss a stall with a fast v_des.
            error = self.target-f.position + (s.kd*self.command_velocity +
                (self.assist.value if s.velocity_feedforward else 0.0))/s.kp
            stopped = abs(f.velocity) <= s.stopped_velocity
            error_threshold = min(s.tracking_error, s.contact_nm/s.kp) if s.velocity_feedforward else s.tracking_error
            candidate = stopped and abs(error) >= error_threshold
            contact = (candidate and closing*(desired-f.position) > .02
                       and closing*error > 0 and closing*self._filtered_torque >= s.contact_nm)
            self._contact_since = (now if self._contact_since is None else self._contact_since) if contact else None
            self._stall_since = (now if self._stall_since is None else self._stall_since) if candidate else None
            if self._contact_since is not None and now-self._contact_since >= s.confirm_s:
                self.mode, self.reason = "contact_hold", "closing_resistance"
                # Convert existing effort to a bounded fixed position spring;
                # never increase the grip effort simply by entering hold.
                equivalent = self.target + (self.assist.value+s.kd*self.command_velocity)/s.kp
                self.command_velocity = 0.0
                self.anchor = self._bounded(equivalent, f, lower, upper, s.hold_nm)
                self.pause()
            elif self._stall_since is not None and now-self._stall_since >= s.stall_s:
                self.block(f, desired, "unconfirmed_motion_stall")

        if self.locally_held:
            budget = s.hold_nm if self.mode == "contact_hold" else min(.5, s.hold_nm)
            self.target = self._bounded(self.anchor, f, lower, upper, budget)
            self.assist.reset()
            return self._wire_command(f, lower, upper, 0.0, budget)

        if active:
            self.target, self.velocity = position_trajectory_step(
                self.target, self.velocity, desired, step_dt, speed, s.acceleration,
                s.braking_acceleration)
        if self.target < lower-1e-8 or self.target > upper+1e-8:
            # Geometry takes priority over a dynamically unreachable reference.
            self.target = max(lower, min(upper, self.target))
            self.velocity = 0.0
        self.command_velocity = self.velocity if active and s.velocity_feedforward else 0.0
        if (fast_follow and active and s.velocity_feedforward and not self._terminal_hold
                and abs(self.target-desired) <= 1e-8 and abs(desired-f.position) > .0075):
            # Bounded final approach, not an unbounded integral or a new endpoint.
            # Helps static friction without repeated stop/restart FF pulses when
            # the reference is complete but the measured jaw still lags it.
            remaining = abs(desired-f.position)
            approach = min(.3, speed, math.sqrt(2*s.braking_acceleration*max(0., remaining-.005)))
            self.command_velocity = math.copysign(approach, desired-f.position)
        proposed = self.target
        self.target = self._bounded(proposed, f, lower, upper, s.command_nm)
        # Saturation projects the reference position back into the effort
        # envelope every sample, preventing position windup. Do not drop v_des
        # abruptly here: that would turn acceleration into a braking pulse.
        at_goal = abs(self.target-desired) <= 1e-8
        if not at_goal or abs(f.position-desired) > .01:
            self._terminal_hold = False
        elif abs(f.position-desired) <= .0075 and (not s.velocity_feedforward or abs(f.velocity) <= s.stopped_velocity):
            self._terminal_hold = True
        braking = s.velocity_feedforward and abs(f.velocity) > s.stopped_velocity and (
            self.command_velocity*f.velocity <= 0 or abs(self.command_velocity) < abs(f.velocity))
        ff = self.assist.update(self.target, f, step_dt, kp=s.kp, kd=s.kd,
            budget=s.command_nm, speed=speed, active=active and not self._terminal_hold and not braking,
            velocity_desired=self.command_velocity)
        return self._wire_command(f, lower, upper, ff, s.command_nm)


class DMMotor:
    """DM MIT packet encoder and feedback decoder."""
    @staticmethod
    def float_to_uint(x: float, x_min: float, x_max: float, bits: int) -> int:
        """Convert float value to unsigned integer with given bit width."""
        if bits <= 0:
            raise ValueError("bits must be positive")

        span = x_max - x_min
        if span <= 0:
            raise ValueError("x_max must be greater than x_min")

        # Clamp
        x = max(min(x, x_max), x_min)

        max_val = (1 << bits) - 1

        # Handle boundaries explicitly to avoid floating point errors
        if x == x_min:
            return 0
        if x == x_max:
            return max_val

        # Linear mapping with rounding
        result = (x - x_min) * max_val / span
        return int(round(result))

    @staticmethod
    def uint_to_float(x: int, x_min: float, x_max: float, bits: int) -> float:
        """Decode the unsigned fixed-width value used by the MIT protocol."""
        if bits <= 0:
            raise ValueError("bits must be positive")
        if x_max <= x_min:
            raise ValueError("x_max must be greater than x_min")
        maximum = (1 << bits) - 1
        if x < 0 or x > maximum:
            raise ValueError(f"value does not fit in {bits} bits")
        return float(x) * (x_max - x_min) / maximum + x_min

    @staticmethod
    def unpack_feedback(
        data: Sequence[int],
        *,
        received_at: float,
        protocol: GripperProtocol | str = GripperProtocol.DM,
        p_range: tuple[float, float] = (-12.5, 12.5),
        v_range: tuple[float, float] = (-10.0, 10.0),
        t_range: tuple[float, float] = (-12.0, 12.0),
    ) -> MotorFeedback:
        """Decode the documented eight-byte motor response payload."""
        if len(data) < 8:
            raise ValueError("motor feedback payload must contain eight bytes")
        values = [int(value) for value in data[:8]]
        if any(value < 0 or value > 0xFF for value in values):
            raise ValueError("motor feedback payload contains a non-byte value")
        protocol = GripperProtocol(protocol)
        position_raw = (values[1] << 8) | values[2]
        velocity_raw = (values[3] << 4) | (values[4] >> 4)
        torque_raw = ((values[4] & 0x0F) << 8) | values[5]
        motor_id = values[0] & 0x0F
        status: Optional[int] = (values[0] >> 4) & 0x0F
        mos_temperature_c = float(values[6])
        motor_temperature_c = float(values[7])
        return MotorFeedback(
            motor_id=motor_id,
            status=status,
            position=DMMotor.uint_to_float(position_raw, *p_range, 16),
            velocity=DMMotor.uint_to_float(velocity_raw, *v_range, 12),
            torque=DMMotor.uint_to_float(torque_raw, *t_range, 12),
            mos_temperature_c=mos_temperature_c,
            motor_temperature_c=motor_temperature_c,
            received_at=float(received_at),
        )

    @staticmethod
    def fmaxf(x: float, y: float) -> float:
        return max(x, y)

    @staticmethod
    def fminf(x: float, y: float) -> float:
        return min(x, y)

    @staticmethod
    def fmaxf3(x: float, y: float, z: float) -> float:
        return max(x, y, z)

    @staticmethod
    def fminf3(x: float, y: float, z: float) -> float:
        return min(x, y, z)

    @staticmethod
    def limit_norm(x: float, y: float, limit: float) -> tuple[float, float]:
        """Scale vector (x, y) length to be <= limit."""
        norm = math.sqrt(x * x + y * y)
        if norm > limit and norm > 0:
            x = x * limit / norm
            y = y * limit / norm
        return x, y

    @staticmethod
    def zero() -> list[int]:
        data = [0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFE]
        return data

    @staticmethod
    def enter_motor_mode() -> list[int]:
        data = [0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFC]
        return data

    @staticmethod
    def exit_motor_mode() -> list[int]:
        data = [0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFD]
        return data

    @staticmethod
    def pack_cmd(
        joint: JointControl,
        p_range: tuple[float, float],
        v_range: tuple[float, float],
        t_range: tuple[float, float],
        kp_range: tuple[float, float],
        kd_range: tuple[float, float],
    ) -> list[int]:
        """Pack motor command into 8 bytes.

        Equivalent to the C function:
        pack_cmd(uint8_t *data, joint_control *joint, ...)
        """
        p_min, p_max = p_range
        v_min, v_max = v_range
        t_min, t_max = t_range
        kp_min, kp_max = kp_range
        kd_min, kd_max = kd_range

        p_des = min(max(p_min, joint.p_des), p_max)
        v_des = min(max(v_min, joint.v_des), v_max)
        kp = min(max(kp_min, joint.kp), kp_max)
        kd = min(max(kd_min, joint.kd), kd_max)
        t_ff = min(max(t_min, joint.t_ff), t_max)

        p_int = DMMotor.float_to_uint(p_des, p_min, p_max, 16)
        v_int = DMMotor.float_to_uint(v_des, v_min, v_max, 12)
        kp_int = DMMotor.float_to_uint(kp, kp_min, kp_max, 12)
        kd_int = DMMotor.float_to_uint(kd, kd_min, kd_max, 12)
        t_int = DMMotor.float_to_uint(t_ff, t_min, t_max, 12)

        data = [0] * 8
        data[0] = (p_int >> 8) & 0xFF
        data[1] = p_int & 0xFF
        data[2] = (v_int >> 4) & 0xFF
        data[3] = ((v_int & 0x0F) << 4) | ((kp_int >> 8) & 0x0F)
        data[4] = kp_int & 0xFF
        data[5] = (kd_int >> 4) & 0xFF
        data[6] = ((kd_int & 0x0F) << 4) | ((t_int >> 8) & 0x0F)
        data[7] = t_int & 0xFF
        return data

    @staticmethod
    def build_motor_packet(
        bus,
        canid,
        data: list[int],
        *,
        frame_flags: int = CAN_CLASSIC_FLAGS,
    ):
        frame_flags = int(frame_flags)
        if frame_flags not in SUPPORTED_CAN_FRAME_FLAGS:
            raise ValueError(
                "gripper frame_flags must be 0 (classic CAN), "
                "4 (CAN-FD), or 5 (CAN-FD+BRS)"
            )
        values = [int(value) for value in data]
        if any(value < 0 or value > 0xFF for value in values):
            raise ValueError("motor packet data contains a non-byte value")
        maximum_length = 8 if frame_flags == CAN_CLASSIC_FLAGS else 64
        if len(values) > maximum_length:
            raise ValueError(
                f"motor packet length {len(values)} exceeds the "
                f"{maximum_length}-byte frame limit"
            )
        packet_type = getattr(
            bxiMsg, "CANFDPacket", getattr(bxiMsg, "CanfdPacket", None)
        )
        if packet_type is None:
            raise RuntimeError("communication.msg.CANFDPacket is unavailable")
        packet = packet_type()
        packet.bus = int(bus)
        packet.frame.can_id = int(canid)
        packet.frame.flags = frame_flags
        packet.frame.len = len(values)
        try:
            packet.frame.data = values
            return packet
        except Exception:
            pass

        padded = values + [0] * (64 - len(values))
        try:
            packet.frame.data = padded
            return packet
        except Exception:
            pass

        for index, value in enumerate(values):
            packet.frame.data[index] = value
        return packet
