"""Read-only session preflight; recorder contracts remain explicit and immutable.

Never sends CAN, enables a motor, discovers hardware or rewrites controller YAML.
Selects only the known DM/BXI identities reported by calibrated live telemetry.
"""
import json
import math

from real_robot_collection.gripper_identity import FIELDS, identities, validate_identities
from real_robot_collection.wrist_diagnostic import convert_raw_source


class AutoIdentityGate:
    MAX_AGE = .3

    def __init__(self):
        self.selected = None
        self.source_at = self.received_at = None
        self.confirmations = 0
        self.error = 'no calibrated live gripper telemetry'

    def reject(self, reason):
        self.selected = None
        self.source_at = self.received_at = None
        self.confirmations = 0
        self.error = str(reason)

    def accept(self, data, now):
        try:
            raw = json.loads(data)
            stamp = float(raw['monotonic'])
            if not math.isfinite(now) or not 0 <= now-stamp <= self.MAX_AGE:
                raise ValueError('stale/future gripper telemetry')
            pair = validate_identities([{k: item[k] for k in FIELDS} for item in raw['samples']])
            allowed = identities(pair[0]['protocol'], pair[1]['protocol'],
                left_bus=pair[0]['bus'], right_bus=pair[1]['bus'])
            if pair != allowed:
                raise ValueError('unreviewed gripper identity; do not infer arbitrary IDs')
            convert_raw_source(raw, pair)
            if self.source_at is not None and stamp <= self.source_at:
                raise ValueError('duplicate/reversed gripper source clock')
            continuous = (pair == self.selected and self.received_at is not None
                          and 0 <= now-self.received_at <= self.MAX_AGE)
            self.confirmations = self.confirmations+1 if continuous else 1
            self.selected, self.source_at, self.received_at = pair, stamp, now
            self.error = None
            return True
        except (ValueError, KeyError, TypeError, OverflowError) as exc:
            self.reject(exc)
            return False

    def arguments(self, now, publisher_count):
        if publisher_count != 1:
            raise ValueError('require exactly one /sonic/gripper_sample publisher')
        if (self.selected is None or self.confirmations < 2 or not math.isfinite(now)
                or not 0 <= now-self.source_at <= self.MAX_AGE
                or not 0 <= now-self.received_at <= self.MAX_AGE):
            raise ValueError('wait for two fresh calibrated gripper samples; '+str(self.error))
        left, right = self.selected
        return ('--left', left['protocol'], '--right', right['protocol'],
                '--left-id', str(left['motor_id']), '--right-id', str(right['motor_id']),
                '--left-bus', str(left['bus']), '--right-bus', str(right['bus']))
