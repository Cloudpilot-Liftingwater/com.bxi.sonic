"""Synthetic offline fixtures only; no DDS, CAN, camera or motor startup."""
import copy
import importlib
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock
import numpy as np
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'collector')]
from configure import configuration, camera_command
from check_config import validate_config
from verify_saved import verify
from real_robot_collection.source_sync import nearest, choose_bundle, packed_pair, AnchorBuffer
from real_robot_collection.synchronization import StateSample, CommandSample, SonicInferenceSample, ImuSample, GripperSample, ImageSample, JsonSample
from real_robot_collection.pico import PicoSnapshot, PackedSnapshot
from real_robot_collection.wrist_diagnostic import convert_raw, validate_raw_row, enabled, WRISTS
from real_robot_collection.gripper_identity import identities


def fixture(index=0, stamp=100.):
    mono=stamp-90
    z=np.zeros(31); mask=np.ones(31,bool)
    raw=dict(schema=1,monotonic=mono-.005,stamp=dict(sec=int(stamp-.005),nanosec=int(((stamp-.005)%1)*1e9)),
             training_eligible=True,ineligible_reasons=[],samples=[])
    for identity in identities():
        raw['samples'].append(dict(**identity,
            open_rad=5.,closed_rad=0.,position_rad=2.5,feedback_monotonic=mono-.01,closure_fraction=.5,
            command_closure_fraction=.8,trigger=.8,trigger_monotonic=mono-.01,
            command=dict(position=1.,sent_monotonic=mono-.01)))
    state=StateSample(mono,stamp-.01,index,z,z,z,z,z,mask,mask,mask)
    command=CommandSample(mono,stamp-.01,index,z,z,z,z,z,mask,np.zeros(29),np.zeros((10,29)))
    inference=SonicInferenceSample(mono,stamp-.009,index,'elf3_sonic_step28800_action_post_fsq',
        configuration()['motion_token']['sha256'],z,np.zeros(64),'live_reference')
    imu=ImuSample(mono,stamp-.008,np.array([1,0,0,0]),np.zeros(3),np.zeros(3),'imu')
    gripper=GripperSample(mono,stamp-.005,np.array([2.5,2.5]),raw)
    images={name:[ImageSample(mono+(.035 if name=='head_rgb' else .126),stamp,index,
        np.zeros((1080,1920,3) if name=='head_rgb' else (480,640,3),np.uint8),'rgb8',name)] for name in WRISTS}
    pico=PicoSnapshot(dict(frame_index=np.array([index]),timestamp_realtime=np.array([stamp-.01]),
        stream_mode=np.array([1]),calibration_ready=np.array([True]),left_trigger=np.array([.8]),right_trigger=np.array([.8])),mono)
    smpl=PackedSnapshot(dict(frame_index=np.array([index]),source_ready=np.array([True]),
        source_stream_mode=np.array([1]),source_calibration_ready=np.array([True])),mono+.001)
    histories=dict(state=[state],command=[command],inference=[inference],imu=[imu],gripper=[gripper])
    return images['head_rgb'][0],histories,images,pico,smpl


def select(f):
    anchor,hist,images,pico,smpl=f
    return choose_bundle(anchor,hist,images,{n:-1 for n in WRISTS},-1,[pico],[smpl])


class DummyNode:
    def __init__(self,name):
        self.subscriptions={}; self.subscription_options={}; self.publications=[]; self.logger=mock.Mock()
    def create_subscription(self,msgtype,topic,callback,qos,**kwargs):
        self.subscriptions[topic]=callback
        self.subscription_options[topic]=(qos,kwargs)
    def create_publisher(self,msgtype,topic,qos): self.publications.append(topic); return mock.Mock()
    def create_timer(self,*args): pass
    def get_logger(self): return self.logger


def load_node():
    messages=types.ModuleType('communication.msg')
    for name in ('ActuatorCmds','ActuatorStates','MotionCommands','TouchSensor'):
        setattr(messages,name,type(name,(),{}))
    with mock.patch.dict(sys.modules,{'communication':types.ModuleType('communication'),'communication.msg':messages}), \
         mock.patch('rclpy.node.Node',DummyNode):
        return importlib.import_module('real_robot_collection.source_node')


class SyncTests(unittest.TestCase):
    def test_config_three_rgb_dm_and_verified_roles(self):
        cfg=configuration(); validate_config(cfg,formal=False,allow_placeholders=False)
        self.assertTrue(enabled(cfg)); self.assertEqual(cfg['metadata']['physical_ports'],{'left':7,'right':1})
        self.assertEqual(cfg['cameras']['head_rgb']['width'],1920)
        self.assertFalse(cfg['cameras']['head_depth']['enabled'])
        with self.assertRaises(ValueError): validate_config(cfg,formal=True,allow_placeholders=False)

    def test_no_fake_training_mapping(self):
        cfg=configuration(); cfg['gripper']['mapping']['verified']=True
        with self.assertRaises(ValueError): enabled(cfg)

    def test_canonical_camera_command_only(self):
        self.assertIn('__ns:=/bxi_trial/left',camera_command('left','/dev/video0'))
        with self.assertRaises(ValueError): camera_command('left','/dev/v4l/by-path/camera')

    def test_buffer_waits_for_late_wrists(self):
        anchor,*_=fixture(); buf=AnchorBuffer(); buf.reset(99.9)
        self.assertIsNone(buf.ready([anchor],100.2))
        self.assertIs(buf.ready([anchor],100.3),anchor)
        self.assertIsNone(buf.consume(anchor,100.3)); self.assertIsNone(buf.ready([anchor],100.3))

    def test_no_prestart_or_too_late_or_clock_reversal(self):
        anchor,*_=fixture(); buf=AnchorBuffer(); buf.reset(100.1)
        self.assertIsNone(buf.ready([anchor],100.3))
        self.assertEqual(buf.consume(anchor,101.),'source_frame_too_late')
        self.assertEqual(buf.consume(anchor,100.3),'non_increasing_source_clock')

    def test_source_not_receipt_time_selects_image(self):
        f=fixture(); anchor,hist,images,pico,smpl=f
        old=ImageSample(10.02,99.9,-1,images['left_wrist_rgb'][0].data,'rgb8','old')
        images['left_wrist_rgb'].append(old)
        bundle,missing=select(f)
        self.assertEqual(missing,[])
        self.assertEqual(bundle['rgb']['left_wrist_rgb'].stamp_unix,100.)
        self.assertGreater(bundle['rgb']['left_wrist_rgb'].received_monotonic,10.1)

    def test_duplicate_camera_frame_is_rejected(self):
        a,h,images,p,s=fixture()
        _,missing=choose_bundle(a,h,images,{n:0 for n in WRISTS},-1,[p],[s])
        self.assertIn('left_wrist_rgb',missing)

    def test_missing_stream_not_fabricated(self):
        a,h,images,p,s=fixture(); h['imu']=[]; images['right_wrist_rgb']=[]
        _,missing=choose_bundle(a,h,images,{n:-1 for n in WRISTS},-1,[p],[s])
        self.assertIn('imu',missing); self.assertIn('right_wrist_rgb',missing)

    def test_mismatched_token_action_rejected(self):
        f=fixture(); f[1]['inference'][0].target_position[0]=1
        # Fixture shares zero arrays: replace command rather than changing them in place.
        c=f[1]['command'][0]
        from dataclasses import replace
        f[1]['command']=[replace(c,position=np.zeros(31))]
        self.assertIn('matched_sonic_inference',select(f)[1])

    def test_repeated_inference_rejected(self):
        a,h,images,p,s=fixture()
        _,missing=choose_bundle(a,h,images,{n:-1 for n in WRISTS},0,[p],[s])
        self.assertIn('matched_sonic_inference',missing)

    def test_pico_smpl_source_frame_join(self):
        a,h,im,p,s=fixture()
        self.assertEqual(packed_pair([p],[s],100.)[2],99.99)
        s.fields['frame_index']=np.array([9])
        self.assertIsNone(packed_pair([p],[s],100.)[0])

    def test_pico_future_or_paused_rejected(self):
        _,_,_,p,s=fixture(); p.fields['timestamp_realtime']=np.array([100.1])
        self.assertIsNone(packed_pair([p],[s],100.)[0])
        p.fields['timestamp_realtime']=np.array([99.99]);p.fields['stream_mode']=np.array([0])
        self.assertIsNone(packed_pair([p],[s],100.)[0])

    def test_raw_dm_identity_and_receipt_freshness(self):
        raw=fixture()[1]['gripper'][0].raw
        np.testing.assert_array_equal(convert_raw(raw,10.)[0],[2.5,2.5])
        with self.assertRaises(ValueError): convert_raw(raw,10.3)
        raw['samples'][1]['motor_id']=1
        with self.assertRaises(ValueError): convert_raw(raw,10.)

    def test_outside_camera_tolerance_rejected(self):
        from dataclasses import replace
        f=fixture();f[2]['right_wrist_rgb']=[replace(f[2]['right_wrist_rgb'][0],stamp_unix=100.1)]
        self.assertIn('right_wrist_rgb',select(f)[1])

    def test_causal_state_and_finite_stamps(self):
        a,h,*_=fixture()
        self.assertIsNone(nearest(h['state'],99.98,.03,causal=True))
        from dataclasses import replace
        self.assertIsNone(nearest([replace(h['state'][0],stamp_unix=float('nan'))],100.,1.))


class NodeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.mod=load_node()

    def node(self,temp):
        cfg=configuration();cfg['dataset']['output_root']=temp
        # Legacy source-clock fixture, independent of the current site default.
        cfg['metadata']['capture_profile'] = 'head30'
        cfg['metadata']['storage_backend'] = 'legacy_jpeg'
        cfg['metadata']['sample_selection'] = 'legacy_source_match'
        cfg['dataset']['fps'] = 30
        path=Path(temp)/'trial.yaml';path.write_text(yaml.safe_dump(cfg))
        with mock.patch('real_robot_collection.node.PicoSubscriber'),mock.patch('real_robot_collection.node.smpl_subscriber'):
            return self.mod.SourceCollectionNode(path,'SYNTHETIC_TEST_ONLY')

    def test_no_motor_publishers_no_depth_and_three_cameras(self):
        with tempfile.TemporaryDirectory() as temp:
            n=self.node(temp)
            try:
                self.assertEqual(n.active_rgb,WRISTS)
                self.assertEqual(n.publications,['/elf3/recording_status'])
                self.assertIn('/sonic/gripper_sample',n.subscriptions)
                qos,opts=n.subscription_options['/sonic/gripper_sample']
                self.assertEqual(qos.depth,32)
                self.assertIs(opts['callback_group'],n.gripper_callback_group)
                self.assertIn('/hardware/head_depth_camera/color/image_raw',n.subscriptions)
                self.assertFalse(n.depth_enabled)
                import inspect
                self.assertEqual(len(inspect.signature(n.subscriptions['/bxi_trial/left/image_raw']).parameters),1)
            finally:n.close()

    def test_clock_jump_is_latched(self):
        with tempfile.TemporaryDirectory() as temp:
            n=self.node(temp)
            try:
                n.clock_offset=90.
                with mock.patch.object(self.mod.time,'time',return_value=100.1),mock.patch.object(self.mod.time,'monotonic',return_value=10.):
                    self.assertFalse(n._clock_ok())
                with mock.patch.object(self.mod.time,'time',return_value=100.),mock.patch.object(self.mod.time,'monotonic',return_value=10.):
                    self.assertFalse(n._clock_ok())
            finally:n.close()

    def test_real_node_raw_callback_keeps_receipt_and_source(self):
        with tempfile.TemporaryDirectory() as temp:
            n=self.node(temp)
            try:
                n.clock_offset=90.
                raw=fixture()[1]['gripper'][0].raw
                with mock.patch.object(self.mod.time,'monotonic',return_value=10.001):
                    n._on_native_gripper(types.SimpleNamespace(data=json.dumps(raw)))
                self.assertEqual(n.gripper_states[-1].received_monotonic,10.001)
                self.assertEqual(n.gripper_states[-1].raw['monotonic'],9.995)
            finally:n.close()

    def test_buffered_capture_tick_and_no_repeated_anchor(self):
        with tempfile.TemporaryDirectory() as temp:
            n=self.node(temp)
            try:
                f=fixture(1,100.05); a,h,images,p,s=f
                for attr,key in (('states','state'),('commands','command'),('imus','imu'),('gripper_states','gripper'),('sonic_inferences','inference')):
                    getattr(n,attr).extend(h[key])
                for name,items in images.items():n.images[name].extend(items)
                n.pico.after.return_value=[p];n.smpl.after.return_value=[s]
                status=JsonSample(9.95,99.95,{'current':{'name':'com.bxi.sonic/sonic_teleop'}})
                n.status_history.append(status)
                n.recorder.start_episode(dict(task_id='UNIT_TEST',instruction_text='SYNTHETIC ONLY'),runtime_metadata={})
                n.source_buffer.reset(100.);n.clock_offset=90.
                with mock.patch.object(self.mod.time,'time',return_value=100.2),mock.patch.object(self.mod.time,'monotonic',return_value=10.2):
                    n._capture_tick();self.assertEqual(n.recorder.frame_count,0)
                with mock.patch.object(self.mod.time,'time',return_value=100.31),mock.patch.object(self.mod.time,'monotonic',return_value=10.31):
                    n._capture_tick();n._capture_tick()
                n.recorder.queue.join()
                self.assertEqual(n.recorder.frame_count,1)
                self.assertEqual(n.last_missing,[])
            finally:n.close()
            self.assertTrue(list((Path(temp)/'SYNTHETIC_TEST_ONLY/three_rgb_gripper_source/_interrupted').glob('*/frames.jsonl')))

    def test_finish_drains_buffer_before_finalizing(self):
        with tempfile.TemporaryDirectory() as temp:
            n=self.node(temp)
            try:
                n.recorder=mock.Mock(recording=True,frame_count=100,meta={})
                n.capture_end_unix=150.;n.finish_deadline=70.
                n.pico.after.return_value=[]
                n.state_machine.phase=n.state_machine.RECORDING
                with mock.patch.object(self.mod.time,'time',return_value=100.),mock.patch.object(self.mod.time,'monotonic',return_value=10.):
                    n._finish('test_finish')
                    n.recorder.end_episode.assert_not_called()
                    self.assertEqual(n.capture_end_unix,100.)
                n.recorder.end_episode.return_value={'frames':100,'path':'synthetic'}
                with mock.patch.object(self.mod.time,'monotonic',return_value=10.41):n._control_tick()
                n.recorder.end_episode.assert_called_once()
            finally:n.recorder.recording=False;n.close()

    def test_abort_preserves_incomplete_episode(self):
        with tempfile.TemporaryDirectory() as temp:
            n=self.node(temp)
            try:
                n.state_machine.start()
                n.recorder.start_episode(dict(task_id='UNIT_TEST',instruction_text='SYNTHETIC ONLY'),runtime_metadata={})
                n._abort('test_abort')
                partial=list((Path(temp)/'SYNTHETIC_TEST_ONLY/three_rgb_gripper_source/_interrupted').glob('*/meta.json'))
                self.assertEqual(len(partial),1)
                self.assertFalse(json.loads(partial[0].read_text())['termination']['success'])
            finally:n.close()

    def test_end_to_end_writer_and_all_image_readback_real_time_gaps(self):
        with tempfile.TemporaryDirectory() as temp:
            n=self.node(temp)
            try:
                n.recorder.start_episode(dict(task_id='UNIT_TEST',instruction_text='SYNTHETIC TEST ONLY'),runtime_metadata={})
                origin=100.
                for i in range(100):
                    stamp=origin+i/30+(i//20)/30
                    f=fixture(i,stamp);bundle,missing=select(f)
                    self.assertEqual(missing,[])
                    n.clock_offset=90.
                    n._save_bundle(f[0],stamp-90,bundle,JsonSample(stamp-90-.1,stamp-.1,{'current':{'name':'com.bxi.sonic/sonic_teleop'}}))
                    n.recorder.queue.join()
                n.recorder.meta['synchronization']=n.sync_summary()
                path=Path(n.recorder.end_episode({'success':True},100)['path'])
                result=verify(path)
                self.assertEqual(result['decoded_images'],300)
                self.assertEqual(result['readback'],'PASS')
                self.assertEqual(result['timing_review'],'PASS')
                self.assertLess(result['actual_row_rate_hz'],30)
                meta=json.loads((path/'meta.json').read_text())
                self.assertFalse(meta['training_export_allowed'])
                self.assertNotIn('published_dataset_contract',meta)
                # Later-received invalid events must also be checked on readback.
                meta['synchronization']['gripper_ingest']['invalid_intervals']=[dict(
                    start_unix=99.999,recover_after_unix=99.999,
                    source_stamp_known=True,reason='test_pause')]
                (path/'meta.json').write_text(json.dumps(meta))
                with self.assertRaisesRegex(ValueError,'invalid gripper source interval'):verify(path)
                meta['synchronization']['gripper_ingest']['invalid_intervals']=[]
                (path/'meta.json').write_text(json.dumps(meta))
                # Corruption / missing files may never pass a readback check.
                image=path/'right_wrist_rgb/rgb/000099.jpg'
                image.write_bytes(b'bad JPEG')
                with self.assertRaisesRegex(ValueError,'JPEG'):verify(path)
            finally:n.close()


if __name__=='__main__': unittest.main()
