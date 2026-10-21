#!/usr/bin/env python3
"""One episode across deliberate POSE pauses; faults still end the session."""
import argparse
import json
from pathlib import Path
import shutil
import signal
import threading
import time

import cv2
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.signals import SignalHandlerOptions

from real_robot_collection.node import RealCollectionNode
from real_robot_collection.source_node import SourceCollectionNode
from session_protocol import (atomic_json, read_intent, START_TIMEOUT, MAX_EPISODE_SECONDS,
                              capture_segment, pause_seconds)


class SessionNode(SourceCollectionNode):
    def __init__(self, config, data_id, folder):
        self.session_folder = Path(folder)
        self.pending_start = None
        self.intent_sequence = 0
        self.session_done = False
        self.stop_reason = None
        self.health_check_at = 0.
        self.last_control_poll = 0.
        self.pauses = []
        self.paused = False
        self.recording_started_mono = None
        self.accounted_pause = 0.
        super().__init__(config, data_id)

    def _result(self, reason, outcome='interrupted'):
        self.session_done = True
        atomic_json(self.session_folder/'session_result.json', dict(reason=reason, outcome=outcome))

    def _start(self, task_id):
        if self.session_done or self.recorder.recording or self.stop_reason:
            return
        super()._start(task_id)
        if self.recorder.recording:
            self.recording_started_mono = time.monotonic()
            # Still bound disk/episode growth, but never label rows as 30/50Hz.
            self.capture_end_unix = self.recorder.started_unix + MAX_EPISODE_SECONDS
            self.finish_deadline = time.monotonic() + MAX_EPISODE_SECONDS + .4
            self.finish_reason = 'maximum_episode_600_seconds'
            self.pending_start = None

    def _update_pause_budget(self, now):
        if self.recording_started_mono is not None and not self.stop_reason:
            total = pause_seconds(self.pauses, now, self.recording_started_mono)
            delta = total-self.accounted_pause
            if self.finish_deadline is not None:
                self.finish_deadline += delta
                self.capture_end_unix += delta
            self.accounted_pause = total

    def _wall_pauses(self):
        return [[start+self.clock_offset, None if end is None else end+self.clock_offset]
                for start, end in self.pauses]

    def sync_summary(self, include_intervals=True):
        value = super().sync_summary(include_intervals)
        value['pose_pauses'] = self._wall_pauses()
        value['pause_semantics'] = 'same_episode_real_source_time_segments'
        return value

    def _anchor_allowed(self, anchor):
        return capture_segment(anchor.stamp_unix, self._wall_pauses()) is not None

    def _save_bundle(self, anchor, tick, bundle, status):
        # Latest-value mode suppresses paused inference rows. Legacy profiles
        # additionally reject other source samples across pause boundaries.
        if getattr(self, 'latest_received', False):
            if not self._anchor_allowed(anchor):
                return False
            super()._save_bundle(anchor, tick, bundle, status)
            return True
        pauses = self._wall_pauses()
        segment = capture_segment(anchor.stamp_unix, pauses)
        stamps = [bundle[k].stamp_unix for k in ('state','command','imu','gripper','inference')]
        stamps += [s.stamp_unix for s in bundle['rgb'].values()]
        stamps += [bundle['pico_stamp'], bundle['smpl'].received_monotonic+self.clock_offset]
        raw = bundle['gripper'].raw
        publication_offset = bundle['gripper'].stamp_unix-float(raw['monotonic'])
        for sample in raw['samples']:
            stamps += [float(sample['feedback_monotonic'])+publication_offset,
                       float(sample['command']['sent_monotonic'])+publication_offset]
        if segment is None or any(capture_segment(t, pauses) != segment for t in stamps):
            return False
        super()._save_bundle(anchor, tick, bundle, status)
        return True

    def _segment_for_anchor(self, anchor):
        return capture_segment(anchor.stamp_unix, self._wall_pauses())

    def _capture_tick(self):
        if self.session_done:
            return
        try:
            self._consume_intent(time.monotonic())
        except Exception as exc:
            self._abort('invalid_session_control:'+str(exc))
            return
        super()._capture_tick()

    def _finish(self, reason):
        self.pending_start = None
        if not self.recorder.recording:
            self._result('start_cancelled:'+reason, 'cancelled')
            return
        if not self.stop_reason:
            self.stop_reason = reason
            super()._finish(reason)

    def _abort(self, reason):
        self.pending_start = None
        if self.recorder.recording:
            super()._abort(reason)  # Preserve partial episode; NEVER discard its rows.
        self._result(reason)

    def _discard(self, reason):
        self.pending_start = None
        self.stop_reason = reason
        if self.recorder.recording:
            self.recorder.meta['synchronization'] = self.sync_summary()
            result = self.recorder.preserve_interrupted_episode(
                reason, min_frames=self.min_frames, discarded=True)
            self.state_machine.abort()
            self.finish_deadline = None
            self.get_logger().info('[REC] discarded (retained, excluded): '+result['path'])
        self._result(reason, 'discarded')

    def _consume_intent(self, now):
        value = read_intent(self.session_folder)
        if value['sequence'] <= self.intent_sequence:
            return
        self.intent_sequence = value['sequence']
        if value['monotonic'] > now+.05:
            raise ValueError('control timestamp is in the future')
        history = value.get('pauses', [])
        # Snapshot retains every boundary even if pause/resume happen between polls.
        for i, (start, end) in enumerate(self.pauses):
            if i >= len(history) or history[i][0] != start or end is not None and history[i][1] != end:
                raise ValueError('pause history was rewritten')
        was_paused = self.paused
        self.pauses = history
        self.paused = bool(history and history[-1][1] is None)
        self._update_pause_budget(now)
        if was_paused and not self.paused and self.pending_start is not None:
            self.pending_start = now+START_TIMEOUT
        if value['command'] == 'start':
            if not self.stop_reason and not self.recorder.recording:
                self.pending_start = value['monotonic'] + START_TIMEOUT
        elif value['command'] in ('pause','resume'):
            if not self.stop_reason and not self.recorder.recording and self.pending_start is None:
                self.pending_start = now+START_TIMEOUT
        elif value['command'] == 'finish':
            self._finish(value['reason'])
        elif value['command'] == 'discard':
            self._discard(value['reason'])
        else:
            self._abort(value['reason'])

    def _control_tick(self):
        if self.session_done:
            return
        now = time.monotonic()
        if now-self.last_control_poll < .02:
            return
        self.last_control_poll = now
        try:
            self._consume_intent(now)
        except Exception as exc:
            self._abort('invalid_session_control:'+str(exc))
            return
        # The supervisor is the sole PICO button consumer. The public ROS
        # fallback can stop/abort but cannot create an unrequested episode.
        with self.lock:
            events = tuple(self.manual_events)
            self.manual_events.clear()
        for event in events:
            if event['command'] == 'finish':
                self._finish('ros_control_finish')
            elif event['command'] == 'abort':
                self._abort('ros_control_abort')
        if self.session_done:
            return
        self._update_pause_budget(now)
        if self.pending_start is not None and not self.paused:
            if now >= self.pending_start:
                self._result('start_readiness_timeout:'+','.join(self._ready(now)))
                return
            if not self._ready(now):
                self._start(None)
        if not self.recorder.recording:
            return
        if not self._clock_ok():
            self._abort('source_clock_changed')
            return
        with self.lock:
            state = self.latest_state_machine
        if state is not None and state.value.get('current', {}).get('name') != self._required_state_name():
            self._finish('sonic_state_exit')
        elif state is None or now-state.received_monotonic > 1.:
            self._abort('state_machine_status_lost')
            return
        if now-self.health_check_at >= 2.:
            self.health_check_at = now
            if shutil.disk_usage(self.source_root).free < 1024**3:
                self._abort('low_disk_less_than_1_GiB')
                return
        if self.finish_deadline is not None and now >= self.finish_deadline and (not self.paused or self.stop_reason):
            self.recorder.meta['synchronization'] = self.sync_summary()
            if self.recorder.frame_count < self.min_frames:
                self._abort('insufficient_synchronized_rows:'+self.finish_reason)
                return
            RealCollectionNode._finish(self, self.finish_reason)
            if self.state_machine.phase == self.state_machine.FINISHED:
                self.session_done = True
            else:
                self._abort('episode_finalization_failed')

    def _status(self):
        status = super()._status()
        status['button_session'] = dict(pending_start=self.pending_start is not None,
            done=self.session_done, stop_reason=self.stop_reason,
            paused=self.paused, pause_count=len(self.pauses), same_episode_resume=True,
            max_episode_seconds=MAX_EPISODE_SECONDS, limit_excludes_pose_pause=True)
        status['operator_hints'] = ['A+X切出POSE暂停，切回继续同一条；左握把+A开始/保存，左握把+B弃用。故障不自动恢复。']
        return status

    def _publish_status(self):
        super()._publish_status()
        # This also provides feedback when no ROS terminal is attached.
        atomic_json(self.session_folder/'session_status.json', self._status(), durable=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--data-id', required=True)
    parser.add_argument('--session-dir', type=Path, required=True)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    rclpy.init(args=None, signal_handler_options=SignalHandlerOptions.NO)
    stopping = threading.Event()
    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, lambda *_: stopping.set())
    node = executor = None
    try:
        node = SessionNode(args.config, args.data_id, args.session_dir)
        executor = MultiThreadedExecutor(num_threads=4)
        executor.add_node(node)
        while not stopping.is_set() and rclpy.ok():
            executor.spin_once(timeout_sec=.2)
    finally:
        if executor:
            executor.shutdown()
        if node:
            node.close()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
