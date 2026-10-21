"""Three RGB / DM or BXI collection with latest-value and legacy source profiles."""
from collections import Counter, deque
import json
import signal
import sys
import threading
import time

import cv2
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.signals import SignalHandlerOptions

from .node import RealCollectionNode, _arguments
from .contracts import sonic_projected_gravity
from .source_sync import AnchorBuffer, choose_bundle, latest_received_bundle, LATEST_RGB_PROFILE, CAMERA_MAX_AGE_S
from .gripper_history import GripperHistory, blocked_by_intervals


class SourceCollectionNode(RealCollectionNode):
    def __init__(self, config_path, data_id):
        super().__init__(config_path, data_id)
        self.latest_received = (self.metadata_cfg.get('capture_profile') == LATEST_RGB_PROFILE
                                and self.metadata_cfg.get('sample_selection') == 'latest_received_v1')
        self.source_buffer = (AnchorBuffer(wait_s=0, max_age_s=float('inf'))
                              if self.latest_received else AnchorBuffer())
        self.latest_rgb = self.metadata_cfg.get('capture_profile') == LATEST_RGB_PROFILE
        self.clock_offset = time.time()-time.monotonic()
        self.clock_failed = False
        self.gripper_history = GripperHistory(self.history_size, .75,
            expected=self.gripper_cfg['expected_identities'])
        self.status_history = deque(maxlen=512)
        self.sync_rejections = Counter()
        self.anchors_seen = 0
        self.capture_end_unix = float('inf')
        self.finish_deadline = None
        self.finish_reason = 'trial_30_seconds'

    def _clock_ok(self):
        if abs(time.time()-time.monotonic()-self.clock_offset) > .02:
            self.clock_failed = True
        return not self.clock_failed

    def _on_state_machine(self, msg):
        super()._on_state_machine(msg)
        if self.latest_state_machine is not None:
            with self.lock:
                self.status_history.append(self.latest_state_machine)

    def _on_native_gripper(self, msg):
        received = time.monotonic()
        with self.lock:
            samples, reason = self.gripper_history.ingest(msg.data, received, self.clock_offset)
            if samples is not None:
                self.gripper_states.append(samples[0])
                self.gripper_commands.append(samples[1])
        if reason:
            self._set_error('invalid buffered gripper sample: '+reason)

    def _ready(self, now):
        missing = super()._ready(now)
        with self.lock:
            latest = max(self.gripper_states, key=lambda s:s.stamp_unix, default=None)
            if (latest is None or not 0 <= now-float(latest.raw['monotonic']) <= .75
                    or blocked_by_intervals(latest.stamp_unix, now+self.clock_offset,
                                            self.gripper_history.invalid_intervals)):
                missing.append('valid_gripper_source_history')
        if not self._clock_ok():
            missing.append('clock_jump_restart_collector')
        for name in self.active_rgb:
            if self.images[name] and now-self.images[name][-1].received_monotonic > .5:
                missing.append('fresh_'+name)
            if self.images[name] and not -.02 <= time.time()-self.images[name][-1].stamp_unix <= .75:
                missing.append('valid_source_clock_'+name)
        return sorted(set(missing))

    def _start(self, task_id):
        was_recording = self.recorder.recording
        super()._start(task_id)
        if not was_recording and self.recorder.recording:
            start = self.recorder.started_unix
            self.source_buffer.reset(start)
            self.sync_rejections.clear()
            self.anchors_seen = 0
            if self.latest_rgb:
                self.last_selected_camera_sequences = {n:-1 for n in self.images}
            with self.lock:
                self.gripper_history.reset_diagnostics()
            self.capture_end_unix = start+30.
            self.finish_deadline = time.monotonic()+30.+self.source_buffer.wait_s+.15
            self.finish_reason = 'trial_30_seconds'

    def _finish(self, source):
        if not self.recorder.recording:
            return super()._finish(source)
        # Drain already captured source frames, rather than trimming the slow
        # wrists by ending immediately when a save request arrives.
        self.capture_end_unix = min(self.capture_end_unix, time.time())
        self.finish_deadline = min(self.finish_deadline, time.monotonic()+.4)
        self.finish_reason = source

    def _control_tick(self):
        super()._control_tick()
        if self.recorder.recording and self.finish_deadline is not None and time.monotonic() >= self.finish_deadline:
            self.recorder.meta['synchronization'] = self.sync_summary()
            if self.recorder.frame_count < self.min_frames:
                self.recorder.preserve_interrupted_episode('insufficient_synchronized_rows', min_frames=self.min_frames)
                self.state_machine.abort()
                self._set_error('trial incomplete: insufficient synchronized rows; data preserved, inspect missing')
            else:
                super()._finish(self.finish_reason)
            self.finish_deadline = None

    def _abort(self, source):
        if self.state_machine.abort() and self.recorder.recording:
            self.recorder.meta['synchronization'] = self.sync_summary()
            result = self.recorder.preserve_interrupted_episode(source, min_frames=self.min_frames)
            self.finish_deadline = None
            self.get_logger().warning('[REC] aborted trial preserved: '+str(result['path']))

    def sync_summary(self, include_intervals=True):
        with self.lock:
            gripper_ingest = self.gripper_history.summary(include_intervals)
        return dict(method=('sonic_inference_latest_received' if self.latest_received else
                            'sonic_inference_latest_rgb' if self.latest_rgb else
                            'source_header_stamp_delayed_head_anchor'), wait_s=self.source_buffer.wait_s,
                    camera_tolerance_s=None if self.latest_received else CAMERA_MAX_AGE_S if self.latest_rgb else .025, robot_tolerance_s=None if self.latest_received else .03,
                    anchors_seen=self.anchors_seen, rejected=dict(self.sync_rejections),
                    no_image_reuse=not self.latest_rgb,
                    camera_max_age_s=None if self.latest_received else CAMERA_MAX_AGE_S if self.latest_rgb else .025,
                    no_interpolation=True, no_timestamp_offset_added=True,
                    pico_smpl_join=('independent_latest_packets_smpl_receipt_clock' if self.latest_received else
                                    'exact_last_frame_index_with_pico_source_realtime'),
                    sample_selection='latest_received_v1' if self.latest_received else 'legacy_source_match',
                    state_machine_time_basis='collector_receive_monotonic',
                    clock_jump=self.clock_failed, gripper_ingest=gripper_ingest)

    def _capture_tick(self):
        if not self.recorder.recording:
            return
        # Detect an alive-but-silent camera as well as launcher-detected exits.
        # Latest-value mode reuses images; a completely stopped camera still aborts.
        now_mono = time.monotonic()
        with self.lock:
            silent = [n for n in self.active_rgb if not self.images[n]
                      or now_mono-self.images[n][-1].received_monotonic > 2.]
        if silent:
            self._abort('camera_input_timeout:'+','.join(silent))
            return
        if not self._clock_ok():
            self.last_missing = ['clock_jump_restart_collector']
            self._set_error('wall/monotonic offset changed >20ms; no further rows accepted')
            return
        # Bounded catch-up; each row keeps a real source stamp, never manufactured idx/fps.
        for _ in range(4):
            now = time.time()
            with self.lock:
                anchor = (self.source_buffer.ready_inference(tuple(self.sonic_inferences), now)
                          if self.latest_rgb else
                          self.source_buffer.ready(tuple(self.images['head_rgb']), now))
                if anchor is None or anchor.stamp_unix > self.capture_end_unix:
                    return
                if not self._anchor_allowed(anchor):
                    self.source_buffer.consume(anchor, now)
                    continue
                # Copy references under a short lock. Input parsing and image writing
                # execute independently; immutable samples are never edited here.
                take = (lambda values: (max(values, key=lambda s: s.received_monotonic),)
                        if values else ()) if self.latest_received else tuple
                histories = {key: take(values) for key, values in (
                    ('state', self.states), ('command', self.commands), ('imu', self.imus),
                    ('gripper', self.gripper_states))}
                histories.update(gripper_invalid_intervals=tuple(self.gripper_history.invalid_intervals),
                                 inference=tuple(self.sonic_inferences))
                images = {n: take(self.images[n]) for n in self.active_rgb}
                statuses = tuple(self.status_history)
            if self.latest_received:
                bundle, missing = latest_received_bundle(anchor, histories, images,
                    self.pico.after(0), self.smpl.after(0), self.clock_offset)
                status = statuses[-1] if statuses else None
            else:
                bundle, missing = choose_bundle(anchor, histories, images,
                    self.last_selected_camera_sequences, self.last_recorded_inference_sequence,
                    self.pico.after(0), self.smpl.after(0), latest_rgb=self.latest_rgb)
                tick = anchor.stamp_unix-self.clock_offset
                status = max((s for s in statuses if 0 <= tick-s.received_monotonic <= .3),
                             key=lambda s: s.received_monotonic, default=None)
            tick = anchor.stamp_unix-self.clock_offset
            if status is None or status.value.get('current', {}).get('name') != self._required_state_name():
                missing.append('sonic_state')
            if not self.latest_received and bundle['smpl'] is not None and not self._smpl_ready(bundle['smpl']):
                missing.append('smpl_ref_ready')
            if missing and self.latest_received:
                # Startup-only missing input: retain the inference for the next tick.
                self.last_missing = sorted(set(missing))
                return
            reason = self.source_buffer.consume(anchor, now)
            self.anchors_seen += 1
            if reason or missing:
                self._reject(([reason] if reason else []) + missing)
                continue
            try:
                if self._save_bundle(anchor, tick, bundle, status) is False:
                    continue
            except Exception as exc:
                self._set_error('source capture rejected: '+str(exc))
                self._reject(['capture_or_writer_failure'])
                if self.latest_received:
                    self._abort('capture_or_writer_failure:'+str(exc))
                    return
                continue
            self.last_recorded_inference_sequence = bundle['inference'].inference_sequence
            self.last_selected_camera_sequences.update({n: s.sequence for n, s in bundle['rgb'].items()})
            self.last_missing = []
            if not self.first_row_logged:
                self.get_logger().info('[REC] first row saved; selection='+('latest_received_v1' if self.latest_received else 'legacy_source_match'))
                self.first_row_logged = True

    def _reject(self, reasons):
        self.last_missing = sorted(set(reasons))
        self.sync_rejections.update(self.last_missing)
        self.dropped_control_ticks += 1

    def _anchor_allowed(self, anchor):
        return True

    def _segment_for_anchor(self, anchor):
        return 0

    def _save_bundle(self, anchor, tick, bundle, status):
        state, command, imu = (bundle[k] for k in ('state', 'command', 'imu'))
        gripper, inference = bundle['gripper'], bundle['inference']
        pico, smpl, rgb = bundle['pico'], bundle['smpl'], bundle['rgb']
        raw = gripper.raw
        stamps = dict(actuator_state=state.stamp_unix, actuator_command=command.stamp_unix,
                      imu=imu.stamp_unix, sonic_inference=inference.stamp_unix,
                      gripper_state=gripper.stamp_unix, gripper_command=gripper.stamp_unix,
                      pico=bundle['pico_stamp'], smpl_ref=(smpl.received_monotonic+self.clock_offset if self.latest_received else bundle['pico_stamp']),
                      **{n: s.stamp_unix for n, s in rgb.items()})
        for item in raw['samples']:
            for suffix, mono in (('feedback',item['feedback_monotonic']),('command',item['command']['sent_monotonic'])):
                stamps[item['side']+'_gripper_'+suffix] = gripper.stamp_unix + (float(mono)-float(raw['monotonic']))
        receives = dict(actuator_state=state.received_monotonic, actuator_command=command.received_monotonic,
                        imu=imu.received_monotonic, sonic_inference=inference.received_monotonic,
                        gripper=gripper.received_monotonic, pico=pico.received_monotonic,
                        smpl_ref=smpl.received_monotonic, **{n:s.received_monotonic for n,s in rgb.items()})
        self.recorder.capture(
            anchor_sequence=anchor.sequence,
            observation={k:getattr(state,k) for k in ('position','velocity','effort','driver_temperature',
                'motor_temperature','joint_present','driver_temperature_present','motor_temperature_present')},
            action={k:getattr(command,k) for k in ('position','velocity','torque','kp','kd','joint_present')},
            effective_continuous_action_29=command.diagnostic_action,
            diagnostic_action_history=command.diagnostic_action_history,
            motion_token=inference.motion_token,
            sonic_inference=dict(sequence=inference.inference_sequence, model_id=inference.model_id,
                model_sha256=inference.model_sha256, policy_status=inference.policy_status,
                target_position=inference.target_position.tolist()),
            projected_gravity=sonic_projected_gravity(imu.quaternion_wxyz, state.position[:29]),
            imu=dict(orientation_wxyz=imu.quaternion_wxyz.tolist(), angular_velocity=imu.angular_velocity.tolist(),
                     linear_acceleration=imu.linear_acceleration.tolist(), frame_id=imu.frame_id),
            rgb_images={n:s.data for n,s in rgb.items()}, depth_image=None, depth_encoding=None,
            pico=pico.json_safe_fields(), smpl_ref=smpl.json_safe_fields(), state_machine=status.value,
            motion_command=None,
            auxiliary=dict(anchor_stamp_unix=anchor.stamp_unix, control_tick_monotonic=tick,
                capture_segment=self._segment_for_anchor(anchor),
                selection_monotonic=time.monotonic(), clock_offset_s=self.clock_offset,
                gripper_ingest_method='source_history_v2',
                gripper_publication_to_callback_s=gripper.received_monotonic-float(raw['monotonic']),
                source_delta_to_anchor_s={k:v-anchor.stamp_unix for k,v in stamps.items()},
                received_monotonic=receives,
                selected_camera_sequences={n:s.sequence for n,s in rgb.items()},
                camera_reused={n:s.sequence == self.last_selected_camera_sequences[n]
                               for n,s in rgb.items()},
                camera_source_age_s={n:anchor.stamp_unix-s.stamp_unix for n,s in rgb.items()},
                pico_smpl_frame_index=pico.frame_index,
                sample_selection='latest_received_v1' if self.latest_received else 'legacy_source_match',
                smpl_timestamp_basis=('collector_receive_monotonic_plus_clock_offset' if self.latest_received
                                      else 'derived_from_exact_pico_frame_index'),
                gripper_feedback_time_basis='CAN_receive_monotonic_bridged_by_same_sample_publish_clocks',
                state_machine_time_basis='collector_receive_monotonic'),
            gripper=dict(coordinate='motor_rad_diagnostic_only', native_raw_sample=raw,
                mapping={'verified':False}, trigger_intent=list(pico.triggers) if pico.triggers else None,
                state=gripper.values.tolist(), state_is_measured=True,
                command=[s['command']['position'] for s in raw['samples']], command_is_final=True),
            source_timestamps=stamps,
            source_receive_offsets={k+'_minus_anchor':v-tick for k,v in receives.items()})

    def _status(self):
        result = super()._status()
        result['synchronization'] = self.sync_summary(include_intervals=False)
        result['three_rgb_gripper_diagnostic_only'] = result.pop('wrist_diagnostic_only')
        result['gripper_identities'] = self.gripper_history.expected
        return result

    def close(self):
        if not self.closed and self.recorder.recording:
            self.recorder.meta['synchronization'] = self.sync_summary()
        super().close()


def main():
    args = _arguments(sys.argv[1:])
    cv2.setNumThreads(1)
    # Keep ROS context alive until callbacks and file writers finish. Humble's
    # default SIGINT handler can invalidate it while end_episode is still busy.
    rclpy.init(args=None, signal_handler_options=SignalHandlerOptions.NO)
    stop_requested = threading.Event()
    previous_handlers = {s:signal.getsignal(s) for s in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP)}
    for signum in previous_handlers:
        signal.signal(signum, lambda *_: stop_requested.set())
    node, executor = None, None
    try:
        node = SourceCollectionNode(args.config, args.data_id)
        executor = MultiThreadedExecutor(num_threads=4)
        executor.add_node(node)
        while not stop_requested.is_set() and rclpy.ok():
            executor.spin_once(timeout_sec=.2)
    except KeyboardInterrupt:
        pass
    finally:
        if executor:
            executor.shutdown()  # Wait for any in-progress finalization; do not race destroy_node.
        if node:
            node.close()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


if __name__ == '__main__':
    main()
