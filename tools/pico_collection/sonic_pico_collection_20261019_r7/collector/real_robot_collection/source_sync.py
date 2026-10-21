"""Latest-received selection and legacy source-clock matching; no IO or control commands."""
from types import SimpleNamespace
import math
import numpy as np
from .gripper_history import blocked_by_intervals

LATEST_RGB_PROFILE = "sonic50_latest_rgb"
CAMERA_MAX_AGE_S = .1

RGB = ('head_rgb', 'left_wrist_rgb', 'right_wrist_rgb')


def stamp_valid(stamp):
    return math.isfinite(stamp) and stamp > 0


def nearest(samples, stamp, tolerance, *, causal=False, after_sequence=-1):
    candidates = [s for s in samples if stamp_valid(s.stamp_unix)
                  and abs(s.stamp_unix-stamp) <= tolerance + 1e-7
                  and (not causal or s.stamp_unix <= stamp + 1e-7)
                  and getattr(s, 'sequence', after_sequence+1) > after_sequence]
    return min(candidates, key=lambda s: (abs(s.stamp_unix-stamp), s.stamp_unix), default=None)


def scalar(snapshot, key):
    values = np.asarray(snapshot.fields[key]).reshape(-1)
    value = float(values[-1])
    if not math.isfinite(value):
        raise ValueError('invalid packed stream scalar '+key)
    return value


def packed_pair(picos, smpls, anchor, tolerance=.2):
    # SMPL has no independent wall-clock stamp. Bind it to the exact PICO
    # frame_index, retain both raw payloads, and declare that derivation.
    for pico in sorted(picos, key=lambda p: p.received_monotonic, reverse=True):
        try:
            stamp = scalar(pico, 'timestamp_realtime')
            if not 0 <= anchor-stamp <= tolerance or not pico.calibrated or pico.stream_mode != 1:
                continue
            matches = [s for s in smpls if s.frame_index == pico.frame_index
                       and abs(s.received_monotonic-pico.received_monotonic) <= tolerance]
            if matches:
                return pico, min(matches, key=lambda s: abs(s.received_monotonic-pico.received_monotonic)), stamp
        except (KeyError, ValueError, IndexError, TypeError):
            continue
    return None, None, None


def choose_bundle(anchor, histories, images, last_camera, last_inference, picos, smpls,
                  camera_tolerance=.025, robot_tolerance=.03, latest_rgb=False):
    stamp = anchor.stamp_unix
    bundle, missing = {}, []
    for key in ('state', 'command', 'imu', 'gripper'):
        bundle[key] = nearest(histories[key], stamp, robot_tolerance, causal=True)
        if bundle[key] is None:
            missing.append(key)
    # Official exporter keeps the latest image until the camera updates.
    # Here source stamps also prevent selecting future images or unbounded stale reuse.
    bundle['rgb'] = {} if latest_rgb else {'head_rgb': anchor}
    for name in RGB if latest_rgb else RGB[1:]:
        sample = nearest(images[name], stamp,
                         CAMERA_MAX_AGE_S if latest_rgb else camera_tolerance,
                         causal=latest_rgb,
                         after_sequence=-1 if latest_rgb else last_camera[name])
        if latest_rgb and sample is not None and sample.sequence < last_camera[name]:
            sample = None
        bundle['rgb'][name] = sample
        if sample is None:
            missing.append(name)
    if latest_rgb:
        target = anchor.inference
        commands = [c for c in histories['command']
                    if abs(c.stamp_unix-stamp) <= robot_tolerance
                    and np.allclose(c.position[:29], target.target_position[:29], rtol=0, atol=1e-5)]
        bundle['command'] = min(commands, key=lambda c: abs(c.stamp_unix-stamp), default=None)
        missing = [key for key in missing if key != 'command']
        if bundle['command'] is None:
            missing.append('command')
    command = bundle['command']
    inference = None
    if command is not None:
        candidates = [s for s in ([anchor.inference] if latest_rgb else histories['inference'])
                      if s.inference_sequence > last_inference
                      and abs(s.stamp_unix-command.stamp_unix) <= .03
                      and abs(s.stamp_unix-stamp) <= robot_tolerance
                      and s.policy_status == 'live_reference'
                      and np.allclose(s.target_position[:29], command.position[:29], rtol=0, atol=1e-5)]
        inference = min(candidates, key=lambda s: abs(s.stamp_unix-command.stamp_unix), default=None)
    bundle['inference'] = inference
    if inference is None:
        missing.append('matched_sonic_inference')
    if command is not None and command.diagnostic_action_history is None:
        missing.append('action_history')
    gripper = bundle['gripper']
    if blocked_by_intervals(gripper.stamp_unix if gripper is not None else None,
                            stamp, histories.get('gripper_invalid_intervals', ())):
        missing.append('gripper_invalid_source_interval')
    if gripper is not None:
        raw = gripper.raw
        for item in raw['samples']:
            for mono in (item['feedback_monotonic'], item['command']['sent_monotonic']):
                source = gripper.stamp_unix + (float(mono)-float(raw['monotonic']))
                if not math.isfinite(source) or abs(source-stamp) > .05:
                    missing.append('gripper_feedback_or_command_source_age')
    pico, smpl, pico_stamp = packed_pair(picos, smpls, stamp)
    bundle.update(pico=pico, smpl=smpl, pico_stamp=pico_stamp)
    if pico is None:
        missing.append('pico_smpl_source_frame_pair')
    return bundle, missing


def latest_received_bundle(anchor, histories, images, picos, smpls, clock_offset):
    """ROS adapter of upstream latest_proprio/latest_image caching, no age join.

    Each inference is still real and saved once. Independently arriving samples
    retain their own source/receipt clocks; this is NOT a synchronous sensor tuple.
    """
    def latest(values):
        return max(values, key=lambda x: x.received_monotonic, default=None)
    bundle = {k: latest(histories[k]) for k in ('state', 'command', 'imu', 'gripper')}
    bundle['inference'] = anchor.inference
    bundle['rgb'] = {name: latest(images[name]) for name in RGB}
    bundle['pico'], bundle['smpl'] = latest(picos), latest(smpls)
    missing = [k for k in ('state', 'command', 'imu', 'gripper', 'pico', 'smpl')
               if bundle[k] is None]
    missing += [name for name, value in bundle['rgb'].items() if value is None]
    if bundle['command'] is not None and bundle['command'].diagnostic_action_history is None:
        missing.append('action_history')
    if bundle['pico'] is not None:
        try:
            bundle['pico_stamp'] = scalar(bundle['pico'], 'timestamp_realtime')
        except (KeyError, TypeError, ValueError, IndexError):
            missing.append('pico_source_timestamp')
    if bundle['smpl'] is not None:
        # SMPL has no source wall clock; do not label a different PICO frame's
        # timestamp as its source time. Keep the raw frame indices in both packets.
        bundle['smpl_stamp'] = bundle['smpl'].received_monotonic + clock_offset
    return bundle, missing


class AnchorBuffer:
    def __init__(self, wait_s=.25, max_age_s=.75):
        if not (wait_s == 0 and max_age_s == float("inf")) and (not .15 <= wait_s <= .5 or not wait_s < max_age_s <= 2):
            raise ValueError('invalid bounded buffering window')
        self.wait_s, self.max_age_s = wait_s, max_age_s
        self.reset(0)

    def reset(self, start_unix):
        self.start_unix = start_unix
        self.last_sequence, self.last_stamp = -1, 0.

    def ready_inference(self, samples, now):
        # No repeat-last-token or synthetic fill: missing inference means a gap.
        anchors = [SimpleNamespace(sequence=s.inference_sequence, stamp_unix=s.stamp_unix,
                                   inference=s) for s in samples
                   if s.policy_status == 'live_reference']
        return self.ready(anchors, now)

    def ready(self, frames, now):
        candidates = [f for f in frames if f.sequence > self.last_sequence
                      and f.stamp_unix >= self.start_unix and now-f.stamp_unix >= self.wait_s]
        return min(candidates, key=lambda f: f.sequence, default=None)

    def consume(self, anchor, now):
        reason = None
        if not stamp_valid(anchor.stamp_unix) or anchor.stamp_unix <= self.last_stamp:
            reason = 'non_increasing_source_clock'
        elif now-anchor.stamp_unix > self.max_age_s:
            reason = 'source_frame_too_late'
        self.last_sequence = anchor.sequence
        self.last_stamp = max(self.last_stamp, anchor.stamp_unix)
        return reason
