"""Read-only collection taps. No CAN writes, model calls, or control decisions."""
from __future__ import annotations

import json
import math
import time


def decode_command(data, profile):
    b = bytes(data)
    if len(b) != 8:
        raise ValueError("MIT command must have 8 bytes")
    def scale(value, limits, bits):
        return limits[0] + value * (limits[1] - limits[0]) / ((1 << bits) - 1)
    return {
        "position": scale((b[0] << 8) | b[1], profile.position_range, 16),
        "velocity": scale((b[2] << 4) | (b[3] >> 4), profile.velocity_range, 12),
        "kp": scale(((b[3] & 15) << 8) | b[4], (0, 500), 12),
        "kd": scale((b[5] << 4) | (b[6] >> 4), (0, 5), 12),
        "feedforward": scale(((b[6] & 15) << 8) | b[7], profile.torque_range, 12),
        "can_data": list(b),
    }


def gripper_snapshot(state, commands, now):
    if not (state._gripper_session_active and state._gripper_armed
            and state._gripper_calibrated and state._gripper_motion_ready
            and not state._gripper_faulted):
        return None
    cycle = getattr(state, '_gripper_control_snapshot', None)
    if cycle is None or not 0 <= now-cycle['monotonic'] <= .05:
        return None
    feedback = cycle['feedback']
    samples = []
    for side, bus in (("left", state._left_bus), ("right", state._right_bus)):
        item, command = feedback.get(bus), commands.get(bus)
        if item is None or command is None:
            return None
        if command['sent_monotonic'] < cycle['monotonic']:
            return None  # must be the commands actually sent for this snapshot
        if not (0 <= now - item.received_at <= .05
                and 0 <= now - command["sent_monotonic"] <= .05):
            return None
        profile = state._gripper_profiles[bus]
        opened, closed = state._soft_limits_for(bus, profile)
        if not all(math.isfinite(x) for x in (opened, closed, item.position,
                                              item.velocity, item.torque)) or abs(closed-opened) < .1:
            return None
        trigger, trigger_at, _ = cycle['inputs'][side]
        samples.append({
            "side": side, "bus": bus, "motor_id": profile.can_id,
            "protocol": profile.protocol.value,
            "profile": "bxi_colleague_rad_v1" if profile.protocol.value == "bxi" else "dm_follow8",
            "position_unit": "rad", "effort_unit": "Nm",
            "position_rad": item.position, "velocity_rad_s": item.velocity,
            "torque_nm": item.torque, "feedback_monotonic": item.received_at,
            "open_rad": opened, "closed_rad": closed,
            "closure_fraction": (item.position-opened)/(closed-opened),
            "command_closure_fraction": (command["position"]-opened)/(closed-opened),
            "trigger": trigger,
            "trigger_monotonic": trigger_at,
            "control": dict(cycle['sides'][bus]),
            "command": dict(command),
        })
    reasons = []
    if cycle['exiting'] or state._gripper_exit_started_at is not None:
        reasons.append('exit_closing')
    if not cycle['pose_allowed'] or state._gripper_pose_paused:
        reasons.append('pose_paused_or_stale')
    if cycle['policy_status'] != 'live_reference' or state.policy.last_status != 'live_reference':
        reasons.append('policy_not_live_reference')
    for s in samples:
        side = s['side']
        if s['trigger_monotonic'] is None:
            reasons.append(f'{side}_trigger_missing')
        elif not 0 <= now-s['trigger_monotonic'] <= state._gripper_trigger_timeout_s:
            reasons.append(f'{side}_trigger_stale')
        if not s['control']['active']:
            reasons.append(f'{side}_control_inactive')
        if s['control']['mode'] == 'blocked_hold':
            reasons.append(f'{side}_blocked_hold')
    return {"schema": 1, "monotonic": now, "training_eligible": not reasons,
            "ineligible_reasons": reasons, "control_monotonic": cycle['monotonic'],
            "coordinate": "raw_motor_rad_and_soft_limit_closure_fraction",
            "samples": samples}


class CollectionTelemetry:
    def __init__(self, node):
        from std_msgs.msg import String
        from rclpy.qos import QoSProfile
        self.message_type = String
        self.node = node
        self.inference = node.create_publisher(String, "/sonic/inference_result", QoSProfile(depth=10))
        self.gripper = node.create_publisher(String, "/sonic/gripper_sample", QoSProfile(depth=10))
        self.reset()

    def reset(self):
        self.commands = {}
        self.sequence = None
        self.last_gripper = -math.inf
        self.next_gripper = None

    def sent(self, bus, data, profile):
        self.commands[bus] = {**decode_command(data, profile), "sent_monotonic": time.monotonic()}

    def emit(self, publisher, value):
        ns = self.node.get_clock().now().nanoseconds
        value["stamp"] = {"sec": ns // 1000000000, "nanosec": ns % 1000000000}
        msg = self.message_type()
        msg.data = json.dumps(value, separators=(",", ":"), allow_nan=False)
        publisher.publish(msg)

    def publish(self, state, output):
        now = time.monotonic()
        if self.next_gripper is None or now+.002 >= self.next_gripper:
            self.last_gripper = now
            # Absolute deadlines: a 19.9/20.1 ms alternation must not turn a
            # requested 50 Hz observer into a 25 Hz one. Never emit catch-up bursts.
            self.next_gripper = (now+.02 if self.next_gripper is None else
                self.next_gripper + max(1, math.floor((now-self.next_gripper)/.02)+1)*.02)
            sample = gripper_snapshot(state, self.commands, now)
            if sample is not None:
                self.emit(self.gripper, sample)
        policy = state.policy
        sequence = getattr(policy, "inference_sequence", None)
        if (not getattr(policy, "motion_token_valid", False)
                or policy.last_status != "live_reference" or sequence is None
                or sequence == self.sequence):
            return
        token = policy.motion_token.tolist()
        if len(token) != 64 or not all(math.isfinite(x) for x in token):
            raise ValueError("invalid SONIC token; collection tap rejected it")
        self.emit(self.inference, {
            "schema": 1, "inference_sequence": int(sequence),
            "model_id": policy.model_id, "model_sha256": policy.model_sha256,
            "joint_names": list(output.joints.layout.names),
            "target_position": output.joints.position.tolist(),
            "motion_token": token, "motion_token_valid": True,
            "policy_status": policy.last_status,
        })
        self.sequence = sequence

    def close(self):
        self.node.destroy_publisher(self.inference)
        self.node.destroy_publisher(self.gripper)
