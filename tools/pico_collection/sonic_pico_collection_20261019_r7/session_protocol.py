"""Same-host private IPC and button edges. No ROS publishers or motor commands."""
import json
import math
import os
from pathlib import Path
import tempfile
import time

from real_robot_collection.pico import scalar

START_TIMEOUT = 30.0
MAX_EPISODE_SECONDS = 600.0


def atomic_json(path, value, *, durable=True):
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix='.'+path.name+'.', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as handle:
            json.dump(value, handle, ensure_ascii=False, allow_nan=False)
            handle.write('\n')
            handle.flush()
            if durable:
                os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def intent(folder, sequence, command, reason, now=None, pauses=()):
    return dict(session=Path(folder).name, sequence=sequence, command=command,
                reason=reason, monotonic=time.monotonic() if now is None else now,
                pauses=[list(p) for p in pauses])


def validate_pauses(pauses, now):
    if not isinstance(pauses, list) or len(pauses) > 4096:
        raise ValueError('invalid pause history')
    previous = -float('inf')
    for i, pair in enumerate(pauses):
        if not isinstance(pair, list) or len(pair) != 2:
            raise ValueError('invalid pause interval')
        start, end = pair
        for value in (start,) if end is None else pair:
            if type(value) not in (int, float) or not math.isfinite(value) or value > now+.05:
                raise ValueError('invalid pause clock')
        if start < previous or end is not None and end < start or end is None and i != len(pauses)-1:
            raise ValueError('unordered pause intervals')
        previous = float('inf') if end is None else end


def pause_seconds(pauses, now, since=-float('inf')):
    return sum(max(0., min(now, now if end is None else end)-max(start, since))
               for start, end in pauses)


def capture_segment(stamp, pauses):
    """None inside a pause, otherwise the numbered uninterrupted section."""
    segment = 0
    for start, end in pauses:
        if stamp < start:
            break
        if end is None or stamp < end:
            return None
        segment += 1
    return segment


def read_intent(folder):
    path = Path(folder)/'intent.json'
    if path.is_symlink() or path.stat().st_size > 1024*1024:
        raise ValueError('invalid control file')
    value = json.loads(path.read_text())
    if (not isinstance(value, dict) or value.get('session') != Path(folder).name
            or type(value.get('sequence')) is not int or value['sequence'] < 1
            or value.get('command') not in ('start', 'pause', 'resume', 'finish', 'discard', 'abort')
            or not isinstance(value.get('reason'), str)
            or len(value['reason']) > 160
            or type(value.get('monotonic')) not in (float, int)
            or not math.isfinite(value['monotonic'])):
        raise ValueError('invalid session control payload')
    validate_pauses(value.get('pauses', []), value['monotonic'])
    return value


class ButtonGate:
    """Require a live released packet before accepting edges, including reconnects."""
    def __init__(self, *, source_clock=True):
        self.source_clock = source_clock
        self.last_received = None
        self.last_source = None
        self.released = False
        self.previous = (False, False)
        self.last_reason = 'no_packet'
        self.last_rejection = None
        self.rejected = 0
        self.events = {}

    def reset(self):
        self.released = False
        self.previous = (False, False)

    def accept(self, packet, now, wall):
        self.last_reason = 'no_edge'
        try:
            event = self._accept(packet, now, wall)
            if event:
                self.events[event] = self.events.get(event, 0) + 1
                self.last_reason = 'accepted:' + event
            return event
        except (ValueError, TypeError, OverflowError) as exc:
            self.last_reason = str(exc)
            self.last_rejection = str(exc)
            self.rejected += 1
            self.reset()
            return None

    def _accept(self, packet, now, wall):
        for key in ('toggle_data_abort', 'toggle_data_collection', 'calibration_ready'):
            if scalar(packet.fields, key, False) not in (False, True, 0, 1):
                raise ValueError('invalid button/calibration flag')
        packet.stream_mode  # Validate before accepting as a live packet.
        source = (float(scalar(packet.fields, 'timestamp_realtime', float('nan')))
                  if self.source_clock else packet.received_monotonic)
        age = now-packet.received_monotonic
        if not math.isfinite(source):
            raise ValueError('invalid_source_clock')
        if not 0 <= age <= .5:
            raise ValueError('receive_age_out_of_range')
        if self.source_clock and not -.05 <= wall-source <= .75:
            raise ValueError('source_clock_stale_or_future')
        if self.last_received is not None and packet.received_monotonic <= self.last_received:
            self.last_reason = 'duplicate_receive'
            return None
        if self.last_source is not None and source <= self.last_source:
            raise ValueError('source_clock_not_advancing')
        if self.last_received is None or packet.received_monotonic-self.last_received > .75:
            self.reset()
        self.last_received, self.last_source = packet.received_monotonic, source
        keys = (packet.abort, packet.toggle_collection)
        if not any(keys):
            self.released = True
        rising = tuple(a and not b for a,b in zip(keys, self.previous))
        self.previous = keys
        if not self.released:
            self.last_reason = 'waiting_for_released_packet'
            return None
        # Official manager_state is the single button authority. POSE is data
        # only; legacy Y+A/Y+B flags must never start or finish a session.
        if self.source_clock:
            return None
        if rising[0]:
            return 'discard'
        if rising[1]:
            return 'toggle'
        return None


def wait_for_session(episode, data, children, folder):
    from trial_lifecycle import child_failure
    deadline = time.monotonic()+MAX_EPISODE_SECONDS+START_TIMEOUT+120
    while True:
        failure = child_failure(children)
        if (episode/'meta.json').is_file():
            return failure
        if (data/'_interrupted').exists():
            return failure or dict(reason='collector_preserved_interrupted_episode')
        if (data/'_discarded').exists():
            return failure
        if failure:
            return failure
        result = Path(folder)/'session_result.json'
        if result.is_file():
            value = json.loads(result.read_text())
            return None if value.get('outcome') in ('cancelled', 'discarded') else value
        try:
            control = read_intent(folder)
        except (OSError, ValueError) as exc:
            return dict(reason='invalid_session_control:'+str(exc))
        now = time.monotonic()
        if now >= deadline + pause_seconds(control.get('pauses', []), now):
            return dict(reason='session_watchdog_timeout')
        time.sleep(.2)
