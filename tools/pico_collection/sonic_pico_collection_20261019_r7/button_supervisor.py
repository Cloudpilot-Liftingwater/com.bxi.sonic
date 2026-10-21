#!/usr/bin/env python3
"""Resident PICO listener. Owns collection children only; never controls the robot."""
import fcntl
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import tempfile
import time
import sys

from configure import configuration, add_gripper_arguments, COLLECTOR_RELEASE
from delivery_paths import runs_root, require_site
from real_robot_collection.pico import PicoSubscriber, PackedSubscriber, PicoSnapshot
from session_protocol import ButtonGate, atomic_json, intent
from sonic_collection_vendor.episode_state import EpisodeState

ROOT = Path(__file__).resolve().parent


class Supervisor:
    def __init__(self, runs, status_file, popen=subprocess.Popen, *, collection_args=(), collection_args_provider=None):
        self.runs, self.status_file = Path(runs), Path(status_file)
        self.popen = popen
        self.collection_args = tuple(collection_args)
        self.collection_args_provider = collection_args_provider
        self.active_collection_args = ()
        self.child = self.log = self.folder = None
        self.sequence = 0
        self.command = None
        self.episode_state = EpisodeState()
        self.phase = 'WAITING_FOR_BUTTON'
        self.last_result = None
        self.last_print_phase = None
        self.last_print_time = 0.
        self.pauses = []
        self.last_manager_at = None
        self.manager_pose = None
        self.pose_return_at = None
        self.inputs = {}
        self.last_input_signature = None
        self.last_input_print = 0.

    def send(self, command, reason, *, boundary=None):
        if self.child is None or self.child.poll() is not None:
            if command in ('finish', 'discard'):
                print('COLLECTION_IGNORED: '+command+'; no active session', flush=True)
            return
        # Finishing cannot be undone by a delayed start or button bounce.
        if self.command in ('finish', 'discard', 'abort'):
            return
        paused = bool(self.pauses and self.pauses[-1][1] is None)
        if command == 'pause':
            if paused:
                return
            self.pauses.append([time.monotonic() if boundary is None else boundary, None])
        elif command == 'resume':
            if not paused:
                return
            self.pauses[-1][1] = time.monotonic() if boundary is None else boundary
        self.sequence += 1
        atomic_json(self.folder/'intent.json', intent(self.folder, self.sequence, command, reason, pauses=self.pauses))
        self.command = command
        if command in ('finish', 'discard', 'abort') and self.episode_state.get_state() == self.episode_state.RECORDING:
            self.episode_state.change_state()
        print('COLLECTION_INTENT='+json.dumps(dict(command=command, reason=reason,
            session=str(self.folder), sequence=self.sequence)), flush=True)
        self.phase = {'finish': 'SAVING', 'discard': 'DISCARDING',
                      'pause': 'PAUSED', 'resume': 'WAITING_FOR_INPUTS'}.get(command, 'PRESERVING_PARTIAL')

    def event(self, command):
        if command == 'toggle':
            state = self.episode_state.get_state()
            if state == self.episode_state.NEED_TO_SAVE:
                print('COLLECTION_BUSY: finalization in progress; wait for completion.', flush=True)
                return
            command = 'start' if state == self.episode_state.IDLE else 'finish'
        if command != 'start':
            self.send(command, 'pico_operator_'+command)
            return
        if self.child is not None:
            print('COLLECTION_BUSY: finish/save/audit must complete before a NEW button press.', flush=True)
            return
        # Resolve BEFORE creating a session or starting any camera/recorder.
        # The child receives fixed explicit identities, never 'auto'.
        selected_args = (tuple(self.collection_args_provider()) if self.collection_args_provider
                         else self.collection_args)
        self.active_collection_args = selected_args
        self.folder = Path(tempfile.mkdtemp(prefix='session_', dir=self.runs))
        self.sequence, self.command = 1, 'start'
        self.pauses = []
        atomic_json(self.folder/'intent.json', intent(self.folder, 1, 'start', 'pico_operator_start'))
        self.log = (self.folder/'session.log').open('x')
        try:
            self.child = self.popen(['bash', str(ROOT/'start.sh'), '--session-dir', str(self.folder), *selected_args],
                stdout=self.log, stderr=subprocess.STDOUT, start_new_session=True)
        except Exception:
            self.log.close()
            self.log = None
            raise
        self.episode_state.change_state()  # Official IDLE -> RECORDING (request accepted).
        self.phase = 'STARTING'
        print('SESSION_GRIPPER_ARGUMENTS='+json.dumps(selected_args), flush=True)
        print(f'COLLECTION_STARTING={self.folder}', flush=True)

    def manager_event(self, packet, gate, now, wall, live_pose_at=None):
        event = gate.accept(packet, now, wall)
        if gate.last_received == packet.received_monotonic:
            self.last_manager_at = packet.received_monotonic
            if packet.stream_mode == 1 and packet.calibrated and not self.manager_pose:
                self.pose_return_at = packet.received_monotonic
            self.manager_pose = packet.stream_mode == 1 and packet.calibrated
            if event == 'discard':
                self.send('discard', 'pico_operator_discard')
            elif event == 'toggle':
                # manager_state has no source timestamp. Never start from that
                # event alone: require a recent validated calibrated POSE too.
                if self.child is not None:
                    self.event('toggle')
                elif self.manager_pose and live_pose_at is not None and 0 <= now-live_pose_at <= .5:
                    self.event('start')
                else:
                    print('START_REJECTED: no fresh calibrated POSE; release then press again.', flush=True)
            if not self.manager_pose:
                self.send('pause', 'pico_left_pose_mode', boundary=packet.received_monotonic)
            elif (live_pose_at is not None and 0 <= now-live_pose_at <= .5
                  and (not self.pauses or live_pose_at >= self.pauses[-1][0])):
                self.send('resume', 'pico_pose_resumed', boundary=packet.received_monotonic)

    def input_health(self, now, last_live):
        # A deliberate POSE exit has a live manager heartbeat even without poses.
        if self.pauses and self.pauses[-1][1] is None:
            if self.last_manager_at is None or now-self.last_manager_at > 1.:
                self.send('abort', 'pico_manager_lost_while_paused')
            elif self.manager_pose and now-max(last_live or 0., self.pose_return_at or 0.) > 1.:
                self.send('abort', 'pico_pose_missing_after_resume')
        elif last_live is None or now-last_live > 1.:
            self.send('abort', 'pico_stream_lost')

    def poll(self):
        if self.child is not None and self.child.poll() is not None:
            code = self.child.returncode
            result_path = self.folder/'launcher_status.json'
            self.last_result = dict(session=str(self.folder), returncode=code,
                                    log=str(self.folder/'session.log'))
            if result_path.is_file():
                try:
                    self.last_result['report'] = json.loads(result_path.read_text())
                except (OSError, ValueError) as exc:
                    # A crashed/failed writer can leave an incomplete report.
                    # Keep its data/logs and keep the listener usable; no retry.
                    self.last_result['report_error'] = str(exc)
                    code = 4
            self.child = None
            self.episode_state.reset_state()
            self.log.close()
            self.log = None
            self.phase = 'WAITING_FOR_BUTTON' if code in (0, 2) else 'ERROR_WAITING_FOR_NEW_BUTTON'
            print('COLLECTION_FINISHED='+json.dumps(self.last_result, ensure_ascii=False), flush=True)

    def publish_status(self):
        value = dict(phase=self.phase, session=str(self.folder) if self.folder else None,
                     last_result=self.last_result, updated_unix=time.time(),
                     listener_pid=os.getpid(), collector_release=COLLECTOR_RELEASE, package_directory=ROOT.name,
                     shortcuts={'left_grip+A':'start / save', 'left_grip+B':'discard (retained, excluded)',
                                'A+X':'pause / resume same episode'},
                     controller_modified=False, automatic_retry=False)
        value['gripper_selection'] = 'controller_telemetry' if self.collection_args_provider else 'explicit'
        value['inputs'] = self.inputs
        value['session_gripper_arguments'] = self.active_collection_args
        if self.child is not None:
            path = self.folder/'session_status.json'
            if path.is_file():
                try:
                    value['collector'] = json.loads(path.read_text())
                except (OSError, ValueError) as exc:
                    value['collector_status_error'] = str(exc)
                    value['collector'] = {}
                state = value['collector']
                if state.get('button_session', {}).get('done'):
                    value['phase'] = 'FINALIZING'
                elif self.command in ('finish', 'discard', 'abort'):
                    pass  # Keep SAVING/DISCARDING/PRESERVING_PARTIAL while paused.
                elif state.get('button_session', {}).get('paused') or self.command == 'pause':
                    value['phase'] = 'PAUSED'
                elif state.get('recording') and self.command in ('start', 'resume'):
                    value['phase'] = 'RECORDING'
                elif state.get('button_session', {}).get('pending_start'):
                    value['phase'] = 'WAITING_FOR_INPUTS'
        atomic_json(self.status_file, value, durable=False)
        now = time.monotonic()
        if value['phase'] != self.last_print_phase or (self.child is not None and now-self.last_print_time >= 5.):
            status = value.get('collector', {})
            print('COLLECTION_STATE='+json.dumps(dict(phase=value['phase'],
                accepted_rows=status.get('accepted_rows', 0), missing=status.get('missing'),
                session=value['session']), ensure_ascii=False), flush=True)
            self.last_print_phase, self.last_print_time = value['phase'], now

    def observe_inputs(self, pico, manager, gate, manager_gate):
        now = time.monotonic()
        self.inputs = {}
        for subscriber, button_gate in ((pico, gate), (manager, manager_gate)):
            packet = subscriber.latest()
            age = now-packet.received_monotonic if packet else None
            try:
                mode, calibrated = (packet.stream_mode, packet.calibrated) if packet else (None, None)
            except (ValueError, TypeError, OverflowError):
                mode, calibrated = None, None
            self.inputs[subscriber.topic] = dict(
                received=subscriber.received, error=subscriber.error,
                age_s=age, fresh=age is not None and 0 <= age <= .5,
                gate_reason=button_gate.last_reason, rejected=button_gate.rejected,
                last_rejection=button_gate.last_rejection, mode=mode, calibrated=calibrated,
                events=dict(button_gate.events))
        signature = tuple((x['fresh'], x['error']) for x in self.inputs.values())
        if signature != self.last_input_signature or now-self.last_input_print >= 10.:
            print('COLLECTION_INPUT='+json.dumps(self.inputs), flush=True)
            self.last_input_signature, self.last_input_print = signature, now

    def close(self):
        if self.child is None:
            return
        if self.child.poll() is None:
            # The launcher owns camera/collector process groups and flushes them.
            try:
                os.killpg(self.child.pid, signal.SIGINT)
            except ProcessLookupError:
                pass
            while self.child.poll() is None:
                try:
                    self.child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    print('Waiting for owned collection to flush; no forced kill.', flush=True)
        self.poll()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_gripper_arguments(parser)
    parser.add_argument('--auto-grippers', action='store_true',
                        help='resolve known DM/BXI pair from fresh controller telemetry on each start')
    args = parser.parse_args()
    if args.auto_grippers and any(a.split('=', 1)[0] in ('--left', '--right', '--left-id', '--right-id', '--left-bus', '--right-bus')
                                 for a in sys.argv[1:]):
        parser.error('--auto-grippers cannot be combined with explicit gripper arguments')
    cfg = configuration(args.left, args.right, args.left_id, args.right_id,
                        left_bus=args.left_bus, right_bus=args.right_bus)
    selected = cfg['gripper']['expected_identities']
    require_site()
    if os.geteuid() == 0:
        raise RuntimeError('listener must run as the operator, never root')
    os.umask(0o077)
    runs = runs_root()
    runs.mkdir(parents=True, exist_ok=True)
    lock = (runs/'listener.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    supervisor = Supervisor(runs, runs/'status.json', collection_args=(
        '--left', args.left, '--right', args.right,
        '--left-id', str(selected[0]['motor_id']), '--right-id', str(selected[1]['motor_id']),
        '--left-bus', str(selected[0]['bus']), '--right-bus', str(selected[1]['bus'])))
    identity_node = None
    if args.auto_grippers:
        import rclpy
        from std_msgs.msg import String
        from rclpy.qos import QoSProfile
        from auto_identity import AutoIdentityGate
        identity_gate = AutoIdentityGate()
        rclpy.init()
        identity_node = rclpy.create_node('elf3_pico_collection_identity_listener')
        identity_node.create_subscription(String, '/sonic/gripper_sample',
            lambda msg: identity_gate.accept(msg.data, time.monotonic()), QoSProfile(depth=10))
        supervisor.collection_args_provider = lambda: identity_gate.arguments(
            time.monotonic(), identity_node.count_publishers('/sonic/gripper_sample'))
        print('EXPECTED_GRIPPERS=auto from verified live telemetry; fixed per episode; no CAN TX.', flush=True)
    else:
        print('EXPECTED_GRIPPERS='+json.dumps(selected), flush=True)
    pico_cfg = cfg['pico']
    pico = PicoSubscriber(pico_cfg['host'], pico_cfg['port'], pico_cfg['topic'])
    gate = ButtonGate()
    # The manager emits official grip+A/B edges even without POSE. A start
    # additionally requires a fresh validated POSE and collector SONIC readiness.
    manager = PackedSubscriber(pico_cfg['host'], pico_cfg['port'], 'manager_state',
        snapshot_type=PicoSnapshot, required_fields=('stream_mode', 'calibration_ready'))
    manager_gate = ButtonGate(source_clock=False)
    stopping = False
    def stop_signal(*_):
        nonlocal stopping
        stopping = True
    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, stop_signal)
    pico.start()
    manager.start()
    cursor, manager_cursor, last_status, last_live, live_pose_at = 0., 0., 0., None, None
    print('PICO_COLLECTION_LISTENER_READY: release buttons; left grip+A starts/saves; left grip+B discards. No motor commands.', flush=True)
    print('COLLECTION_ENDPOINT='+json.dumps(pico_cfg)+'; listener started; check COLLECTION_INPUT for actual messages.', flush=True)
    try:
        while not stopping:
            if identity_node is not None:
                rclpy.spin_once(identity_node, timeout_sec=0.)
            supervisor.poll()
            pose_packets = pico.after(cursor)
            manager_packets = manager.after(manager_cursor)
            # Snapshot queues BEFORE sampling the clock. A subscriber thread may
            # otherwise enqueue a new one-shot button after the old `now` value.
            now = time.monotonic()
            for packet in pose_packets:
                cursor = max(cursor, packet.received_monotonic)
                gate.accept(packet, now, time.time())
                if gate.last_received == packet.received_monotonic:
                    last_live = packet.received_monotonic
                    if packet.stream_mode != 1 or not packet.calibrated:
                        live_pose_at = None
                        # Mode transitions have one authority: manager_state.
                        # Different subscriber queues must not reorder boundaries.
                    else:
                        live_pose_at = packet.received_monotonic
            for packet in manager_packets:
                manager_cursor = max(manager_cursor, packet.received_monotonic)
                try:
                    supervisor.manager_event(packet, manager_gate, now, time.time(), live_pose_at)
                except Exception as exc:
                    print('COLLECTION_REQUEST_FAILED='+str(exc), flush=True)
                    supervisor.phase = 'ERROR_WAITING_FOR_NEW_BUTTON'
            if last_live is None or now-last_live > 1.:
                gate.reset()
                live_pose_at = None
            supervisor.input_health(now, last_live)
            if now-last_status >= 1.:
                supervisor.observe_inputs(pico, manager, gate, manager_gate)
                supervisor.publish_status()
                last_status = now
            time.sleep(.02)
    finally:
        pico.close()
        manager.close()
        supervisor.close()
        if identity_node is not None:
            identity_node.destroy_node()
            rclpy.shutdown()
        supervisor.phase = 'LISTENER_STOPPED'
        supervisor.publish_status()


if __name__ == '__main__':
    main()
