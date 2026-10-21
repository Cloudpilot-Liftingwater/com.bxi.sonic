"""Passive recorder: ROS subscriptions, ZMQ SUB, local observation socket only."""
import argparse
import base64
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import shutil
import signal
import socket
import threading
import time

from tracking_tap import ADDRESS, decode


def safe_json(value):
    if isinstance(value,float) and not math.isfinite(value):return None
    if isinstance(value,dict):return {k:safe_json(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [safe_json(v) for v in value]
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seconds', type=float, default=60.)
    parser.add_argument('--mod-root', type=Path, default=Path(__file__).resolve().parents[3],
                        help='actual loaded SONIC Mod directory, used only for provenance')
    args = parser.parse_args()
    if not 1 <= args.seconds <= 600:
        parser.error('--seconds must be 1..600')
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output/'events.jsonl').exists():
        parser.error('output already contains a recording')
    if shutil.disk_usage(args.output).free < 2*1024**3:
        parser.error('less than 2 GiB free disk space')
    import rclpy
    import zmq
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
    from communication.msg import ActuatorStates, ActuatorCmds
    from std_msgs.msg import String

    tap = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    tap.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4*1024**2)
    try:
        tap.bind(ADDRESS)
    except OSError:
        tap.close()
        raise SystemExit('已有跟踪记录器占用本机观察接口；不会停止或替换它。')
    tap.setblocking(False)
    counts, lost, errors = Counter(), Counter(), Counter()
    stop = threading.Event()
    writer_failed = []
    events = queue.Queue(maxsize=2048)
    rclpy.init()
    node = rclpy.create_node('sonic_tracking_readonly_'+str(os.getpid()), enable_rosout=False)
    qos = QoSProfile(depth=512, reliability=ReliabilityPolicy.BEST_EFFORT,
                     durability=DurabilityPolicy.VOLATILE)
    context = zmq.Context()
    subscribers = []
    started = time.monotonic()
    roots = args.mod_root.expanduser().resolve()
    metadata = dict(schema=1, release='sonic_tracking_20261014_v1', hostname=socket.gethostname(),
        boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        started_wall_ns=time.time_ns(), started_monotonic=started,
        duration_requested_s=args.seconds, ros_environment={k:os.environ.get(k) for k in
            ('ROS_DOMAIN_ID','RMW_IMPLEMENTATION','ROS_LOCALHOST_ONLY','CYCLONEDDS_URI')},
        angle_unit='rad', hardware_commands_sent=0, camera_required=False,
        schema_note='ROS header source times and host receive times both retained; trace carries actual consumed model input',
        fingerprints={})
    for name in ('policy.py','tracking_tap.py','state.py','bxi_demo.py','mod.yaml'):
        path = roots/name
        try:metadata['fingerprints'][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError as exc:metadata['fingerprints'][str(path)] = str(exc)
    (args.output/'metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2)+'\n')

    def submit(kind, value):
        item = dict(kind=kind, received_monotonic=time.monotonic(),
                    received_wall_ns=time.time_ns(), ros_now_ns=node.get_clock().now().nanoseconds,
                    value=value)
        try:events.put_nowait(item);counts[kind] += 1
        except queue.Full:lost[kind] += 1

    def write():
        try:
            with (args.output/'events.jsonl').open('x', buffering=1024*1024) as stream:
                while True:
                    item = events.get()
                    if item is None:break
                    if isinstance(item['value'], bytes):
                        item['value'] = dict(packet_b64=base64.b64encode(item['value']).decode())
                    stream.write(json.dumps(safe_json(item), separators=(',', ':'), allow_nan=False)+'\n')
                stream.flush();os.fsync(stream.fileno())
        except Exception as exc:
            writer_failed.append(str(exc));stop.set()
    worker = threading.Thread(target=write, name='tracking-file-writer', daemon=True)
    worker.start()

    def motor(msg, kind):
        try:
            command = kind == 'command'
            value = dict(stamp_ns=msg.header.stamp.sec*1000000000+msg.header.stamp.nanosec,
                frame_id=msg.header.frame_id,
                names=list(msg.actuators_name if command else msg.name),
                position=list(msg.pos if command else msg.position),
                velocity=list(msg.vel if command else msg.velocity))
            for name in (('kp','kd','torque') if command else ('effort',)):
                value[name] = list(getattr(msg, name))
            submit(kind, value)
        except Exception as exc:errors[kind+': '+str(exc)] += 1
    node.create_subscription(ActuatorStates, '/hardware/actuator_states', lambda msg:motor(msg,'state'), qos)
    node.create_subscription(ActuatorCmds, '/hardware/actuators_cmds', lambda msg:motor(msg,'command'), qos)
    def text_message(msg, kind):
        try:submit(kind, json.loads(msg.data))
        except Exception as exc:errors[kind+': '+str(exc)] += 1
    for topic, kind in (('/hardware/state_machine_info','state_machine'),
                        ('/sonic/inference_result','inference_result')):
        node.create_subscription(String, topic, lambda msg,k=kind:text_message(msg,k), qos)
    poller = zmq.Poller()
    for topic, port in (('pose',5556), ('smpl_ref',5557)):
        sub = context.socket(zmq.SUB)
        sub.setsockopt(zmq.LINGER, 0);sub.setsockopt(zmq.RCVHWM, 512)
        sub.setsockopt(zmq.SUBSCRIBE, topic.encode())
        sub.connect('tcp://127.0.0.1:'+str(port))
        poller.register(sub, zmq.POLLIN);subscribers.append((sub,topic))
    for sig in (signal.SIGINT,signal.SIGTERM):
        signal.signal(sig, lambda *_:stop.set())
    last_status = -1.
    print('记录已开始：只观察，按原流程手动遥操；不需要相机。Ctrl+C 只结束本记录器。', flush=True)
    try:
        while not stop.is_set() and time.monotonic()-started < args.seconds:
            rclpy.spin_once(node, timeout_sec=.001)
            ready = dict(poller.poll(0))
            for sub,topic in subscribers:
                if sub not in ready:continue
                for _ in range(64):
                    try:submit(topic, sub.recv(zmq.NOBLOCK))
                    except zmq.Again:break
            for _ in range(64):
                try:
                    packet = tap.recv(65536)
                    decode(packet)
                    submit('trace', packet)
                except BlockingIOError:break
                except Exception as exc:errors['trace: '+str(exc)] += 1
            elapsed = time.monotonic()-started
            if elapsed-last_status >= 5.:
                last_status = elapsed
                graph = {name:node.count_publishers(name) for name in
                    ('/hardware/actuator_states','/hardware/actuators_cmds','/sonic/inference_result')}
                submit('health', dict(counts=dict(counts), queue_lost=dict(lost), publishers=graph))
                print('TRACKING '+json.dumps(dict(seconds=round(elapsed,1),counts=dict(counts),lost=dict(lost)),ensure_ascii=False),flush=True)
                if shutil.disk_usage(args.output).free < 512*1024**2:
                    errors['disk free below 512 MiB'] += 1;break
    finally:
        # Only our observer resources are closed; no robot control publication.
        tap.close()
        for sub,_ in subscribers:sub.close()
        context.term()
        if worker.is_alive():
            while worker.is_alive():
                try:events.put(None, timeout=.2);break
                except queue.Full:continue
            worker.join(timeout=20.)
        if worker.is_alive():writer_failed.append('writer did not finish within 20 seconds')
        node.destroy_node()
        if rclpy.ok():rclpy.shutdown()
        result = dict(counts=dict(counts),queue_lost=dict(lost),errors=dict(errors),
            writer_errors=writer_failed, elapsed_s=time.monotonic()-started,
            missing=[k for k in ('state','command','trace','pose','state_machine') if counts[k]==0],
            hardware_commands_sent=0)
        (args.output/'capture_result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
        print('CAPTURE_RESULT='+json.dumps(result,ensure_ascii=False),flush=True)
    return 0 if not result['missing'] and not writer_failed and not lost and not errors else 3


if __name__ == '__main__':
    raise SystemExit(main())
