"""Collector-only source validation and invalid intervals; never sends commands."""
from collections import Counter, deque
import json
import math

import numpy as np

from .wrist_diagnostic import convert_raw_source
from .gripper_identity import identities, validate_identities
from .synchronization import GripperSample


def blocked_by_intervals(sample_stamp, anchor_stamp, intervals):
    # A known invalid publication blocks until a strictly newer valid publication.
    # An unlocatable/malformed packet covers the pending window and requires a
    # valid publication newer than its receipt, not a queued pre-error packet.
    return any(b['start_unix'] <= anchor_stamp and
               (sample_stamp is None or sample_stamp <= b['recover_after_unix'])
               for b in intervals)


class GripperHistory:
    def __init__(self, size=2048, max_transport_s=.75, clock_tolerance_s=.02, *, expected=None):
        self.expected = validate_identities(identities() if expected is None else expected)
        self.invalid_intervals = deque(maxlen=size)
        self.max_transport_s = max_transport_s
        self.clock_tolerance_s = clock_tolerance_s
        self.reset_diagnostics()

    def reset_diagnostics(self):
        # Keep source history/barriers across recording start. Only counters reset.
        self.counts = Counter()
        self.rejections = Counter()
        self.ineligible_reasons = Counter()
        self.latencies = deque(maxlen=10000)

    def ingest(self, payload, received, clock_offset):
        stamp, value = None, None
        self.counts['received'] += 1
        try:
            value = json.loads(payload)
            if not isinstance(value, dict):
                raise ValueError('gripper payload must be an object')
            published = float(value['monotonic'])
            raw_stamp = value['stamp']
            sec, nsec = raw_stamp['sec'], raw_stamp['nanosec']
            if type(sec) is not int or type(nsec) is not int or not 0 <= nsec < 1000000000:
                raise ValueError('invalid publication ROS stamp')
            candidate_stamp = sec+nsec*1e-9
            if (not all(math.isfinite(v) for v in (published, received, clock_offset, candidate_stamp))
                    or published < 0 or candidate_stamp <= 0 or received < published
                    or abs(candidate_stamp-published-clock_offset) > self.clock_tolerance_s):
                raise ValueError('invalid publication clock relation')
            stamp = candidate_stamp  # trusted source location, even when ineligible
            latency = received-published
            self.latencies.append(latency)
            if latency > .05:
                self.counts['transport_over_50ms'] += 1
            if latency > self.max_transport_s:
                raise ValueError('transport exceeds buffered window')
            state, command = convert_raw_source(value, self.expected)
            self.counts['accepted'] += 1
            return (GripperSample(received, stamp, state, value),
                    GripperSample(received, stamp, command, value)), None
        except (ValueError, TypeError, KeyError, IndexError, OverflowError) as exc:
            reason = str(exc)
            self.rejections[reason] += 1
            if isinstance(value, dict) and isinstance(value.get('ineligible_reasons'), list):
                self.ineligible_reasons.update(str(r) for r in value['ineligible_reasons'])
            self.counts['rejected'] += 1
            receipt_unix = received+clock_offset
            self.invalid_intervals.append(dict(
                start_unix=stamp if stamp is not None else receipt_unix-self.max_transport_s,
                recover_after_unix=stamp if stamp is not None else receipt_unix,
                source_stamp_known=stamp is not None, reason=reason))
            return None, reason

    def summary(self, include_intervals=True):
        ages = np.asarray(self.latencies, dtype=float)*1000
        latency = {'count':len(ages)}
        if len(ages):
            latency.update(min=float(ages.min()), median=float(np.median(ages)),
                           p95=float(np.percentile(ages,95)), max=float(ages.max()))
        result = dict(method='source_history_v2', counts=dict(self.counts),
                    rejected=dict(self.rejections), ineligible_reasons=dict(self.ineligible_reasons),
                    publication_to_callback_ms=latency, source_freshness_s=.05,
                    max_transport_s=self.max_transport_s, clock_tolerance_s=self.clock_tolerance_s,
                    invalid_interval_count=len(self.invalid_intervals))
        if include_intervals:
            result['invalid_intervals'] = list(self.invalid_intervals)
        return result
