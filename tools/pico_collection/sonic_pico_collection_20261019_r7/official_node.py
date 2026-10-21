#!/usr/bin/env python3
"""Read-only ELF3 ROS transport for the official SONIC collector; never publishes commands."""
import argparse
from collections import Counter
import json
from pathlib import Path
import queue
import signal
import threading
import time

import numpy as np
import yaml
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from communication.msg import ActuatorStates, ActuatorCmds
from sensor_msgs.msg import Image, Imu
from std_msgs.msg import String

from delivery_paths import runs_root, data_root
from official_session import OfficialSession
from session_protocol import ButtonGate
from real_robot_collection.pico import PicoSubscriber, PackedSubscriber, PicoSnapshot
from real_robot_collection.contracts import canonicalize_named, sonic_projected_gravity
from real_robot_collection.synchronization import decode_raw_image, quaternion_multiply_wxyz
from real_robot_collection.gripper_identity import check_sample_identity
from contract import CAMERAS, MODEL_SHA


def stamp(msg):
    return float(msg.header.stamp.sec)+float(msg.header.stamp.nanosec)*1e-9


class Collector(Node):
    def __init__(self, cfg):
        super().__init__('elf3_official_collection')
        self.cfg=cfg
        self.session=OfficialSession(runs_root(),data_root(),cfg)
        self.cache={}
        self.lock=threading.Lock()
        self.queue=queue.Queue(maxsize=32)
        self.failure=None
        self.counts=Counter()
        self.last_inference=None
        self.quit=threading.Event()
        self.groups=[]
        self.subscriptions_owned=[]
        latest=QoSProfile(history=HistoryPolicy.KEEP_LAST,depth=1,reliability=ReliabilityPolicy.BEST_EFFORT)
        reliable=QoSProfile(depth=256)
        def sub(cls,topic,callback,qos=latest):
            group=MutuallyExclusiveCallbackGroup();self.groups.append(group)
            def guarded(msg):
                try:callback(msg)
                except Exception as exc:
                    self.counts['invalid:'+topic]+=1
                    self.get_logger().error(f'{topic}: {exc}',throttle_duration_sec=5)
            self.subscriptions_owned.append(self.create_subscription(cls,topic,guarded,qos,callback_group=group))
        robot=cfg['robot']
        sub(ActuatorStates,robot['actuator_state_topic'],lambda m:self.put('state',self.body(m.position,m.name),stamp(m)))
        sub(ActuatorCmds,robot['actuator_command_topic'],lambda m:self.put('command',self.body(m.pos,m.actuators_name),stamp(m)))
        sub(Imu,robot['imu_topic'],self.imu)
        sub(String,cfg['gripper']['sample_topic'],self.gripper)
        sub(String,cfg['motion_token']['topic'],self.inference,reliable)
        for key,raw in CAMERAS.items():
            sub(Image,cfg['cameras'][raw]['topic'],lambda m,k=key:self.put(k,decode_raw_image(m),stamp(m)))
        self.pico=PicoSubscriber(cfg['pico']['host'],cfg['pico']['port'],'pose')
        self.manager=PackedSubscriber(cfg['pico']['host'],cfg['pico']['port'],'manager_state',snapshot_type=PicoSnapshot)
        self.pico.start();self.manager.start()
        self.gate=ButtonGate(source_clock=False)
        self.manager_at=0.
        self.mode=None
        group=MutuallyExclusiveCallbackGroup();self.groups.append(group)
        self.create_timer(.01,self.buttons,callback_group=group)
        self.worker=threading.Thread(target=self.write_loop,name='official-writer')
        self.worker.start()
        print('PICO_COLLECTION_LISTENER_READY: left grip+A starts/saves; left grip+B discards. No motor commands.',flush=True)

    @staticmethod
    def body(values,names):
        value,_=canonicalize_named(values,names,label='body29')
        return value[:29]

    def put(self,key,value,source_time):
        with self.lock:
            self.cache[key]=(value,float(source_time),time.monotonic())

    def imu(self,msg):
        q=quaternion_multiply_wxyz([msg.orientation.w,msg.orientation.x,msg.orientation.y,msg.orientation.z],
            self.cfg['robot']['imu_orientation_correction_wxyz'])
        if not np.isfinite(q).all() or np.linalg.norm(q)<1e-6:raise ValueError('invalid IMU quaternion')
        self.put('imu',q,stamp(msg))

    def gripper(self,msg):
        raw=json.loads(msg.data)
        samples=raw['samples']
        if len(samples)!=2:raise ValueError('both motor fields required by this training contract')
        measured,command=[],[]
        for sample,expected in zip(samples,self.cfg['gripper']['expected_identities']):
            check_sample_identity(sample,expected)
            measured.append(float(sample['position_rad']))
            command.append(float(sample['command']['position']))
        if not np.isfinite(measured+command).all():raise ValueError('nonfinite motor angle')
        # No 30/50ms freshness cutoff. Retain receipt/source times for offline inspection.
        self.put('gripper',(np.array(measured),np.array(command)),raw['monotonic'])
        self.put('gripper_sources',dict(training_eligible=raw.get('training_eligible'),
            channels=[dict(side=x['side'],feedback_monotonic=x['feedback_monotonic'],
                command_monotonic=x['command']['sent_monotonic']) for x in samples]),raw['monotonic'])

    def enqueue(self,item):
        while not self.quit.is_set():
            try:self.queue.put(item,timeout=.1);return
            except queue.Full:continue  # Backpressure, not put_nowait/drop-and-abort.
        if self.failure:raise RuntimeError(self.failure)

    def inference(self,msg):
        raw=json.loads(msg.data)
        if raw.get('motion_token_valid') is not True or raw.get('model_sha256')!=MODEL_SHA:
            raise ValueError('not a valid token from the configured SONIC model')
        seq=int(raw['inference_sequence']);st=raw['stamp'];source=float(st['sec'])+float(st['nanosec'])*1e-9
        identity=(seq,source)
        if self.last_inference is not None and seq==self.last_inference[0]:return
        # A controller re-entry may reset the sequence; timestamp distinguishes epochs.
        self.last_inference=identity
        token=np.asarray(raw['motion_token'],dtype=np.float32)
        if token.shape!=(64,) or not np.isfinite(token).all():raise ValueError('token must contain 64 finite values')
        received=time.monotonic()
        with self.lock:
            snapshot=dict(self.cache)  # Arrays are replaced, never mutated by callbacks.
        missing=sorted(set(['state','command','imu','gripper',*CAMERAS])-snapshot.keys())
        self.session.missing=missing
        if missing:
            self.counts['waiting_initial_inputs']+=1
            return
        pose=self.pico.latest(); manager=self.manager.latest()
        live=(pose is not None and received-pose.received_monotonic<=.1
              and pose.stream_mode==1 and pose.calibrated
              and manager is not None and received-manager.received_monotonic<=.5
              and manager.stream_mode==1 and raw.get('policy_status')=='live_reference')
        smpl=np.asarray(pose.fields['smpl_pose'],dtype=np.float32).reshape(-1,63)[0].copy() if live else np.zeros(63,dtype=np.float32)
        if smpl.size!=63:raise ValueError(f'Expected official 63 SMPL values, got {smpl.size}')
        body=snapshot['state'][0];grip,gripcmd=snapshot['gripper'][0]
        frame={'observation.state':np.concatenate((body,grip)).astype(np.float32),
            'observation.projected_gravity':sonic_projected_gravity(snapshot['imu'][0],body).astype(np.float32),
            'action.wbc':np.concatenate((snapshot['command'][0],gripcmd)).astype(np.float32),
            'action.motion_token':token,'teleop.smpl_pose':smpl}
        for key in CAMERAS:frame['observation.images.'+key]=snapshot[key][0]
        provenance=dict(inference_sequence=seq,inference_stamp_unix=source,received_monotonic=received,
            source_stamps={k:v[1] for k,v in snapshot.items()},
            receipt_ages_s={k:received-v[2] for k,v in snapshot.items()},
            source_clock_domains={'gripper':'robot_monotonic','other':'ROS_header'},
            live_pose=bool(live),gripper_sources=snapshot.get('gripper_sources',(None,))[0])
        self.enqueue(('frame',(frame,provenance)))

    def buttons(self):
        for packet in self.manager.after(self.manager_at):
            self.manager_at=packet.received_monotonic
            mode=packet.stream_mode==1 and packet.calibrated
            if mode!=self.mode:
                self.enqueue(('command',('pose' if mode else 'pause',packet.received_monotonic)))
                self.mode=mode
            event=self.gate.accept(packet,time.monotonic(),time.time())
            if event:self.enqueue(('command',(event,packet.received_monotonic)))

    def write_loop(self):
        try:
            last_status=0.
            while True:
                try:kind,value=self.queue.get(timeout=.2)
                except queue.Empty:kind,value=None,None
                if kind=='stop':break
                if kind=='frame':self.session.add(*value)
                elif kind=='command':self.session.handle(*value)
                if time.monotonic()-last_status>=1:
                    status=self.session.publish_status()
                    print('COLLECTION_STATE='+json.dumps(dict(phase=status['phase'],accepted_rows=status['accepted_rows'],
                        missing=status['missing'],queue=self.queue.qsize(),counts=self.counts)),flush=True)
                    last_status=time.monotonic()
        except BaseException as exc:
            self.failure=str(exc);self.session.fail(exc);self.quit.set()
        finally:
            try:self.session.close()
            except BaseException as exc:self.failure=str(exc);self.session.fail(exc)

    def close(self):
        self.pico.close();self.manager.close()
        if not self.quit.is_set():self.enqueue(('stop',None))
        self.worker.join()
        self.quit.set()


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);a=p.parse_args()
    rclpy.init();node=Collector(yaml.safe_load(a.config.read_text()))
    executor=MultiThreadedExecutor(num_threads=8);executor.add_node(node)
    try:
        while rclpy.ok() and not node.quit.is_set():executor.spin_once(timeout_sec=.2)
    except KeyboardInterrupt:pass
    finally:
        node.quit.set() if node.failure else None
        executor.shutdown()
        node.close();node.destroy_node()
        if rclpy.ok():rclpy.shutdown()
    if node.failure:raise SystemExit(node.failure)


if __name__=='__main__':main()
