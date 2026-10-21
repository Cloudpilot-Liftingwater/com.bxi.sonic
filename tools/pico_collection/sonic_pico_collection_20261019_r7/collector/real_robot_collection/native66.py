"""Physical-1DoF data contract; no guessed motor-to-training conversion."""
import math
import numpy as np


def validate_mapping(mapping):
    if not isinstance(mapping, dict) or mapping.get("verified") is not True:
        raise ValueError("native66 mapping is not hardware-verified; diagnostic data only")
    for key in ("calibration_id", "evidence", "coordinate_unit"):
        if not isinstance(mapping.get(key), str) or not mapping[key].strip():
            raise ValueError(f"mapping.{key} is required")
    if mapping.get("input_coordinate") != "soft_limit_closure_fraction":
        raise ValueError("mapping input must be soft_limit_closure_fraction")
    for side in ("left", "right"):
        hardware = mapping.get("hardware")
        if hardware is not None:
            identity = hardware.get(side) if isinstance(hardware, dict) else None
            if (not isinstance(identity, dict)
                    or identity.get("protocol") not in ("dm", "bxi")
                    or identity.get("profile") != {"dm": "dm_follow8", "bxi": "bxi_colleague_rad_v1"}[identity["protocol"]]
                    or type(identity.get("bus")) is not int or not 0 <= identity["bus"] <= 255
                    or type(identity.get("motor_id")) is not int or not 0 <= identity["motor_id"] <= 15):
                raise ValueError(f"mapping.hardware.{side} must bind protocol/profile/bus/motor_id")
        curve = np.asarray(mapping.get(side), dtype=float)
        if (curve.ndim != 2 or curve.shape[1] != 2 or len(curve) < 3
                or not np.isfinite(curve).all() or curve[0, 0] != 0 or curve[-1, 0] != 1
                or not np.all(np.diff(curve[:, 0]) > 0)):
            raise ValueError(f"mapping.{side} needs measured [fraction, coordinate] knots including 0, 1")
        dy = np.diff(curve[:, 1])
        if not (np.all(dy > 0) or np.all(dy < 0)):
            raise ValueError(f"mapping.{side} output must be strictly monotonic")


def convert_sample(value, mapping, now):
    validate_mapping(mapping)
    if value.get("schema") != 1 or value.get("training_eligible") is not True:
        raise ValueError("gripper sample is not active human demonstration")
    published = float(value["monotonic"])
    if not 0 <= now-published <= .05:
        raise ValueError("stale publication or different monotonic clock")
    items = value.get("samples", [])
    if len(items) != 2 or [x.get("side") for x in items] != ["left", "right"]:
        raise ValueError("atomic left/right sample required")
    states, commands = [], []
    for item, bus, motor in zip(items, (5, 6), (1, 4)):
        if mapping.get("hardware") is not None:
            expected = mapping["hardware"][item["side"]]
            if any(item.get(k) != expected[k] for k in ("bus", "motor_id", "protocol", "profile")):
                raise ValueError("sample does not match hardware-bound training mapping")
        elif ((item.get("bus"), item.get("motor_id")) != (bus, motor)
              or item.get("protocol", "dm") != "dm"
              or item.get("profile", "dm_follow8") != "dm_follow8"):
            raise ValueError("legacy DM mapping cannot be used for different gripper hardware")
        for t in (item["feedback_monotonic"], item["command"]["sent_monotonic"]):
            if not 0 <= published-float(t) <= .05 or now-float(t) > .05:
                raise ValueError("stale feedback/command; republication cannot refresh it")
        opened, closed = float(item["open_rad"]), float(item["closed_rad"])
        if not all(math.isfinite(x) for x in (opened, closed)) or abs(closed-opened) < .1:
            raise ValueError("invalid soft limits")
        curve = np.asarray(mapping[item["side"]], dtype=float)
        for key, position, result in (
                ("closure_fraction", item["position_rad"], states),
                ("command_closure_fraction", item["command"]["position"], commands)):
            fraction = (float(position)-opened)/(closed-opened)
            if not math.isfinite(fraction) or not math.isclose(fraction, float(item[key]), abs_tol=1e-7):
                raise ValueError("inconsistent closure coordinate")
            if not -.01 <= fraction <= 1.01:
                raise ValueError("outside calibrated range")
            result.append(float(np.interp(fraction, curve[:, 0], curve[:, 1])))
    return np.asarray(states), np.asarray(commands)


def validate_gripper_row(gripper):
    raw = gripper.get("native_raw_sample")
    if not isinstance(raw, dict):
        raise ValueError("native66 requires traceable raw motor sample")
    state, command = convert_sample(raw, gripper.get("mapping"), float(raw["monotonic"]))
    for expected, key in ((state, "state"), (command, "command")):
        actual = np.asarray(gripper.get(key), dtype=float)
        if actual.shape != (2,) or not np.allclose(actual, expected, rtol=0, atol=1e-7):
            raise ValueError(f"native66 {key} does not match raw sample + mapping")
