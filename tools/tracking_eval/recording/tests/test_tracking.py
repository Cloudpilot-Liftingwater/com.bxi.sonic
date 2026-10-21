import ast
import base64
import json
from pathlib import Path
import socket
import struct
import sys
import time
from types import SimpleNamespace as NS
import uuid

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import tracking_tap as tap
import install
from analysis_core import causal_indices, estimate_lag, metrics, named, packed, source_time
from geometry import CONTRACT, JOINTS, RobotGeometry, bend
from analyze import analyze


def snapshot():
    q=np.array(CONTRACT['default_q'],dtype=np.float32)
    model=np.zeros((1,1770),dtype=np.float32)
    model[0,:720]=np.arange(720)*.001
    p=NS(inference_sequence=1,last_status='live_reference',source_blend_active=False,
        latest_live_ref_time=time.monotonic(),model_id='test',model_sha256='a'*64,joint_names=JOINTS,
        default_dof_pos=q.copy(),action_scale=np.array(CONTRACT['action_scale']),yaw_offset=0.,
        target_dof_pos=q.copy(),motion_token=np.zeros(64,dtype=np.float32))
    ref=NS(frame_index=10,sequence=2,root_quat=np.tile([1,0,0,0],(10,1)).astype(np.float32))
    return p,ref,model,q


@pytest.fixture
def observer_socket(monkeypatch):
    address='\0test-sonic-tracking-'+uuid.uuid4().hex
    monkeypatch.setattr(tap,'ADDRESS',address)
    if tap._socket is not None:tap._socket.close()
    tap._socket=None;tap._retry_at=0.
    receiver=socket.socket(socket.AF_UNIX,socket.SOCK_DGRAM);receiver.bind(address);receiver.settimeout(1.)
    yield receiver
    receiver.close()
    if tap._socket is not None:tap._socket.close()
    tap._socket=None;tap._retry_at=0.


def test_actual_unix_snapshot_is_exact_and_arrays_unchanged(observer_socket):
    p,r,m,q=snapshot();before=[a.copy() for a in (m,q,p.target_dof_pos)]
    tap.capture(p,r,'live',m,q,q,q,q,q,time.monotonic())
    meta,arrays=tap.decode(observer_socket.recv(65536))
    np.testing.assert_array_equal(arrays['model_input'],m.reshape(-1))
    for a,b in zip(before,(m,q,p.target_dof_pos)):np.testing.assert_array_equal(a,b)
    assert meta['joint_names']==list(JOINTS)
    assert arrays['decoded_target'].shape==(29,)


def test_absent_recorder_does_not_raise_or_recompute(monkeypatch):
    monkeypatch.setattr(tap,'ADDRESS','\0absent-'+uuid.uuid4().hex)
    tap._socket=None;tap._retry_at=0.
    p,r,m,q=snapshot();tap.capture(p,r,'live',m,q,q,q,q,q,time.monotonic())
    assert tap._socket is None


@pytest.mark.parametrize('error',[BlockingIOError(),OSError('gone'),ValueError('bad sink')])
def test_tap_failure_is_contained(monkeypatch,error):
    class Sink:
        def send(self,data):raise error
        def close(self):pass
    monkeypatch.setattr(tap,'_socket',Sink())
    p,r,m,q=snapshot();tap.capture(p,r,'live',m,q,q,q,q,q,time.monotonic())


def test_truncated_snapshot_rejected(observer_socket):
    p,r,m,q=snapshot();tap.capture(p,r,'live',m,q,q,q,q,q,time.monotonic())
    packet=observer_socket.recv(65536)
    with pytest.raises(ValueError):tap.decode(packet[:-1])


@pytest.mark.parametrize('query,expected',[(.9,-1),(1.,0),(1.02,0),(1.2,-1),(2.,1)])
def test_causal_alignment_never_uses_future_or_stale_command(query,expected):
    assert causal_indices([1.,2.],[query],max_age=.05).tolist()==[expected]


def test_source_clock_uses_header_not_receipt():
    e=dict(value=dict(stamp_ns=10_000_000_000),ros_now_ns=10_020_000_000,received_monotonic=40.)
    assert source_time(e)==pytest.approx(39.98)
    e['value']['stamp_ns']=0
    with pytest.raises(ValueError):source_time(e)


def test_missing_joint_never_becomes_zero():
    x=named([1.],['b'],['a','b']);assert np.isnan(x[0]) and x[1]==1.
    with pytest.raises(ValueError):named([1.,2.],['a','a'],['a'])


def test_metric_definitions():
    m=metrics([1.,-2.,3.,np.nan]);assert m['n']==3
    assert m['mae_deg']==2.
    assert m['rmse_deg']==pytest.approx(np.sqrt(14/3))
    assert metrics([np.nan])['rmse_deg'] is None


def test_known_fifty_ms_lag():
    t=np.arange(0,6,.005);target=20*np.sin(t*4.)
    measured=20*np.sin((t-.05)*4.)
    r=estimate_lag(t,target,measured,np.ones(len(t),dtype=bool))
    assert abs(r['estimated_lag_ms']-50)<=5
    assert r['lag_compensated_rmse_deg']<.3


def test_static_motion_has_no_latency_claim():
    t=np.arange(0,6,.005)
    assert estimate_lag(t,t*0,t*0,np.ones(len(t),bool))['estimated_lag_ms'] is None


def test_long_gaps_do_not_produce_latency_estimate():
    t=np.arange(200)*.1;target=20*np.sin(t)
    assert estimate_lag(t,target,target,np.ones(len(t),bool))['estimated_lag_ms'] is None


def test_latency_never_crosses_clock_epochs():
    t=np.arange(600)*.01;target=20*np.sin(t*4.)
    # Every 200 ms is a new clock segment, shorter than the lag search window.
    result=estimate_lag(t,target,target,np.ones(len(t),bool),epochs=np.arange(600)//20)
    assert result['status']=='insufficient_contiguous_motion'


@pytest.mark.parametrize('points,expected',[([[0,0,0],[1,0,0],[2,0,0]],0),([[0,0,0],[1,0,0],[1,1,0]],90)])
def test_geometric_bend_is_coordinate_independent(points,expected):
    assert bend(points)==pytest.approx(expected)
    transformed=np.asarray(points)[:,[2,0,1]]*3+np.array([10,20,30])
    assert bend(transformed)==pytest.approx(expected)


def test_urdf_bends_use_all_joint_coordinates():
    g=RobotGeometry();q=np.zeros(29);base=g.bends(q)
    q[18]=.5
    changed=g.bends(q)
    assert np.isfinite(base).all()
    assert abs(changed[0]-base[0])>20
    np.testing.assert_allclose(changed[1:],base[1:])


def make_install_tree(tmp_path, baseline='baseline_policy.py.txt'):
    original=(ROOT/baseline).read_bytes()
    for rel in install.FOLDERS:
        folder=tmp_path/rel;folder.mkdir(parents=True);(folder/'policy.py').write_bytes(original)


@pytest.mark.parametrize('baseline',['baseline_policy.py.txt','legacy_policy.py.txt'])
def test_install_plan_idempotent_and_rollback(tmp_path,baseline):
    make_install_tree(tmp_path,baseline);changes=install.plan(tmp_path);assert len(changes)==4
    for path,(_,data) in changes.items():install.write(path,data)
    assert not install.plan(tmp_path)
    for path,(_,data) in install.plan(tmp_path,remove=True).items():install.write(path,data)
    for rel in install.FOLDERS:
        assert (tmp_path/rel/'policy.py').read_bytes()==(ROOT/baseline).read_bytes()
    assert len(install.plan(tmp_path))==4


def test_installer_refuses_unknown_policy(tmp_path):
    make_install_tree(tmp_path)
    (tmp_path/install.FOLDERS[0]/'policy.py').write_text('print("local modification")')
    with pytest.raises(ValueError,match='unknown policy'):install.plan(tmp_path)


@pytest.mark.parametrize('sink',[None,'working','raising'])
def test_real_inference_method_math_unchanged_with_observer(sink):
    source=(ROOT/'baseline_policy.py.txt').read_text()
    result=[]
    for text in (source,install.patched(source)):
        tree=ast.parse(text)
        cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='SonicTeleopPolicy')
        func=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='inference_step')
        def capture(*args):
            if sink=='raising':raise RuntimeError('observer error')
        ns=dict(np=np,NUM_JOINTS=29,MOTION_TOKEN_DIM=64,FSQ_STEP=.0625,ACTION_CLIP=20.,
                _capture_tracking=None if sink is None else capture)
        exec(compile(ast.Module(body=[func],type_ignores=[]),'<actual-policy-method>','exec'),ns)
        raw=np.linspace(-30,30,29,dtype=np.float32)
        p,r,m,q=snapshot()
        p._active_reference=lambda:(r,'live',123.)
        p.head_joint_target=np.zeros(2);r.head_joint_pos=np.zeros((10,2))
        p._begin_source_transition=lambda *_:None
        p._build_model_input=lambda *_:m.copy()
        p.input_buffer=m.copy();p._inputs={}
        p._backend=NS(run=lambda _:dict(action=raw.reshape(1,29),motion_token=np.zeros((1,64))))
        p._blend_source_target=lambda candidate,now:candidate*.75+q*.25
        p._reported_status='live_reference'
        p.live_reference_stale=False;p.active_reference_kind='legacy_window'
        value=ns['inference_step'](p,q,q,np.array([1,0,0,0]),np.zeros(3))
        result.append(value.copy())
    np.testing.assert_array_equal(*result)


def pack_fields(fields):
    header=dict(endian='le',fields=[dict(name=k,dtype='i64' if v.dtype==np.int64 else 'f32',shape=list(v.shape)) for k,v in fields.items()])
    text=json.dumps(header,separators=(',',':')).encode();assert len(text)<1280
    return b'pose'+text.ljust(1280,b'\0')+b''.join(v.tobytes() for v in fields.values())


def synthetic_recording(folder, seconds=6.):
    folder.mkdir(exist_ok=True);events=[];zero=np.array(CONTRACT['default_q'],dtype=np.float32)
    scale=np.array(CONTRACT['action_scale'],dtype=np.float32)
    offset=1_800_000_000.-1000.
    def emit(kind,value,t):
        events.append(dict(kind=kind,value=value,received_monotonic=t+.001,
            received_wall_ns=int((offset+t+.001)*1e9),ros_now_ns=int((offset+t+.001)*1e9)))
    for i,t0 in enumerate(np.arange(0,seconds,.01)):
        t=1000.+t0
        target=zero+np.deg2rad(20*np.sin(4*t0))
        actual=zero+np.deg2rad(20*np.sin(4*(t0-.05))+2.)
        points=np.zeros((24,3),dtype=np.float32)
        for a,b,c in ((16,18,20),(17,19,21),(1,4,7),(2,5,8)):
            points[a]=[0,0,0];points[b]=[1,0,0];points[c]=[1.8,.6,0]
        root=np.array([1,0,0,0],dtype=np.float32);wrists=target[list((19,20,21,26,27,28))]
        model=np.zeros(1770,dtype=np.float32);model[:720]=np.tile(points.reshape(-1),10);model[780:840]=np.tile(wrists,10)
        frames=dict(frame_index=np.array([i],dtype=np.int64),smpl_pose=np.zeros((1,21,3),dtype=np.float32),
            smpl_joints=points[None],body_quat_w=root[None],joint_pos=target[None].astype(np.float32))
        emit('pose',dict(packet_b64=base64.b64encode(pack_fields(frames)).decode()),t)
        meta=dict(schema=1,session='synthetic',pid=123,sequence=i+1,source='live',policy_status='live_reference',
            source_blend_active=bool(t0<.2),reference_frame_index=i,reference_sequence=i,reference_received_monotonic=t,
            started_monotonic=t,completed_monotonic=t+.001,completed_wall_ns=int((offset+t+.001)*1e9),
            model_id='synthetic_test_model',model_sha256='a'*64,joint_names=list(JOINTS),default_q=zero.tolist(),
            action_scale=scale.tolist(),yaw_offset_rad=0.,tap_sent=i,tap_dropped=0,tap_errors=0,tap_previous_max_overhead_ns=0)
        header=json.dumps(meta,separators=(',',':')).encode()
        raw=(target-zero)/scale
        arrays=(model,raw,raw,target,target,actual,np.zeros(29),np.zeros(64),np.tile(root,10))
        packet=tap.MAGIC+struct.pack('<I',len(header))+header+b''.join(np.asarray(a,dtype='<f4').tobytes() for a in arrays)
        emit('trace',dict(packet_b64=base64.b64encode(packet).decode()),t+.001)
        emit('command',dict(stamp_ns=int((offset+t+.002)*1e9),names=list(JOINTS),position=target.tolist(),kp=[30.]*29),t+.002)
        emit('state',dict(stamp_ns=int((offset+t+.003)*1e9),names=list(JOINTS),position=actual.tolist()),t+.003)
        if i%10==0:emit('state_machine',dict(current=dict(name='com.bxi.sonic/sonic_teleop')),t)
    events.sort(key=lambda e:e['received_monotonic'])
    (folder/'events.jsonl').write_text(''.join(json.dumps(e,separators=(',',':'))+'\n' for e in events))
    (folder/'metadata.json').write_text(json.dumps(dict(synthetic=True,description='50ms delay + 2deg bias; no hardware')))
    (folder/'capture_result.json').write_text(json.dumps(dict(missing=[],writer_errors=[],queue_lost={},errors={})))
    return folder


def test_full_recording_to_report(tmp_path):
    folder=synthetic_recording(tmp_path/'synthetic')
    report=analyze(folder)
    assert report['synthetic'] is True
    assert report['quality']['scored_state_rows']>500
    assert report['quality']['source_blended_inferences']==20
    assert report['quality']['human_encoder_matching']['unique_exact_tensor_match']==6000
    row=report['joints'][0]
    assert 40<=row['delay']['estimated_lag_ms']<=65
    assert 1.8<row['delay']['lag_compensated_rmse_deg']<2.2
    assert row['rmse_deg']>row['delay']['lag_compensated_rmse_deg']
    arrays=np.load(folder/'analysis/inference_arrays.npz',allow_pickle=False)
    assert arrays['model_input'].shape==(600,1770)
    assert np.isfinite(arrays['consumed_smpl_rotvec_rad']).all()
    assert '合成测试数据' in (folder/'analysis/report.html').read_text()


def test_no_data_is_not_success(tmp_path):
    (tmp_path/'events.jsonl').write_text('')
    report=analyze(tmp_path)
    assert report['status']=='PARTIAL_OR_NO_VALID_SAMPLES'
    assert all(x['n']==0 for x in report['joints'])


def test_unknown_body_state_cannot_be_scored_as_sonic(tmp_path):
    folder=synthetic_recording(tmp_path/'synthetic',seconds=.5)
    p=folder/'events.jsonl'
    p.write_text(''.join(line for line in p.read_text().splitlines(True) if json.loads(line)['kind']!='state_machine'))
    report=analyze(folder)
    assert report['quality']['scored_state_rows']==0


def test_missing_feedback_joint_stays_missing(tmp_path):
    folder=synthetic_recording(tmp_path/'synthetic',seconds=.5)
    p=folder/'events.jsonl';events=[json.loads(line) for line in p.read_text().splitlines()]
    for e in events:
        if e['kind']=='state':e['value']['names'].pop(0);e['value']['position'].pop(0)
    p.write_text(''.join(json.dumps(e)+'\n' for e in events))
    report=analyze(folder)
    assert report['joints'][0]['n']==0 and report['joints'][1]['n']>0
    assert report['status']=='PARTIAL_OR_NO_VALID_SAMPLES'
    assert report['quality']['invalid_state_components']==50
    assert report['status']=='PARTIAL_OR_NO_VALID_SAMPLES'


def test_nonfinite_gain_is_not_valid_tracking(tmp_path):
    folder=synthetic_recording(tmp_path/'synthetic',seconds=.5)
    p=folder/'events.jsonl';events=[json.loads(line) for line in p.read_text().splitlines()]
    for e in events:
        if e['kind']=='command':e['value']['kp'][0]=float('inf')
    p.write_text(''.join(json.dumps(e)+'\n' for e in events))
    report=analyze(folder)
    assert report['joints'][0]['n']==0 and report['joints'][1]['n']>0


def test_ambiguous_smpl_twist_is_not_invented(tmp_path):
    folder=synthetic_recording(tmp_path/'synthetic',seconds=.5)
    p=folder/'events.jsonl';events=[json.loads(line) for line in p.read_text().splitlines()]
    first=next(e for e in events if e['kind']=='pose')
    fields=packed(base64.b64decode(first['value']['packet_b64']),'pose')
    fields['smpl_pose'][0,0,0]=.3
    fields['frame_index']=np.array([999],dtype=np.int64)
    duplicate=dict(first);duplicate['value']=dict(packet_b64=base64.b64encode(pack_fields(fields)).decode())
    events.insert(1,duplicate)
    p.write_text(''.join(json.dumps(e)+'\n' for e in events))
    report=analyze(folder)
    assert report['quality']['human_encoder_matching']['ambiguous_twist_angles']>=10
    arrays=np.load(folder/'analysis/inference_arrays.npz',allow_pickle=False)
    assert np.isnan(arrays['consumed_smpl_rotvec_rad'][0]).all()


def test_active_robot_blocks_install_without_writes(tmp_path,monkeypatch):
    make_install_tree(tmp_path)
    monkeypatch.setattr(install,'BASE',tmp_path)
    original=install.plan
    monkeypatch.setattr(install,'plan',lambda remove=False:original(tmp_path,remove=remove))
    monkeypatch.setattr(install.socket,'gethostname',lambda:'elf3-test-01')
    monkeypatch.setattr(install.subprocess,'run',lambda *a,**k:NS(returncode=0,stdout='123 hardware_elf3'))
    monkeypatch.setattr(sys,'argv',['install.py','--apply'])
    with pytest.raises(SystemExit,match='正常结束'):install.main()
    assert len(original(tmp_path))==4
    assert not list(tmp_path.parent.glob('sonic_tracking_backup_*'))


def test_resident_services_allow_hook_install(tmp_path,monkeypatch):
    import re
    processes = [
        '1044 python3 /opt/ros/humble/bin/ros2 launch remote_controller remote_controller.launch.py',
        '1923 python3 /opt/ros/humble/bin/ros2 launch robot_gateway gateway.launch.py',
        '1933 /bin/bash /opt/bxi/bxi_rc_ros2/bxi_rc_ros2_start.sh',
        '4219 /home/bxi/bxi_ws/bxi_rl_controller_ros2_example/install/lib/remote_controller/remote_controller',
        '4229 /opt/bxi/bxi_rc_ros2/lib/robot_gateway/ros_bridge_node',
        '5000 python3 -B button_supervisor.py',
    ]
    def pgrep(cmd,**kwargs):
        assert cmd[:2]==['pgrep','-af']
        matches=[p for p in processes if re.search(cmd[2],p)]
        return NS(returncode=0 if matches else 1,stdout='\n'.join(matches))
    make_install_tree(tmp_path)
    monkeypatch.setattr(install,'BASE',tmp_path)
    original=install.plan
    monkeypatch.setattr(install,'plan',lambda remove=False:original(tmp_path,remove=remove))
    monkeypatch.setattr(install.socket,'gethostname',lambda:'elf3-test-01')
    monkeypatch.setattr(install.subprocess,'run',pgrep)
    monkeypatch.setattr(sys,'argv',['install.py','--apply'])
    install.main()
    assert not original(tmp_path)


@pytest.mark.parametrize('process',[
    '334460 python3 /opt/ros/humble/bin/ros2 launch bxi_example_py_elf3 example_demo_hw.launch.py',
    '334482 /opt/bxi/bxi_ros2_pkg/lib/hardware_elf3/hardware_elf3',
    '334484 python3 /home/bxi/bxi_ws/bxi_rl_controller_ros2_example/install/lib/bxi_example_py_elf3/bxi_example_py_elf3_demo',
    '5001 python3 -m bxi_example_py_elf3.framework.runtime.mod_node_runner --node smpl_bridge',
    '5002 python3 pico_manager_thread_server.py --manager',
    '5003 python3 -B session_node.py',
    '5004 python3 -B -u record.py --output /tmp/run',
])
def test_controller_input_and_recording_still_block(process,monkeypatch):
    import re
    def pgrep(cmd,**kwargs):
        matches=bool(re.search(cmd[2],process))
        return NS(returncode=0 if matches else 1,stdout=process if matches else '')
    monkeypatch.setattr(install.subprocess,'run',pgrep)
    with pytest.raises(SystemExit,match='正常结束'):install.require_inactive()


def test_process_probe_error_refuses_install(monkeypatch):
    monkeypatch.setattr(install.subprocess,'run',lambda *a,**k:NS(returncode=2,stdout='',stderr='synthetic error'))
    with pytest.raises(SystemExit,match='process check failed'):install.require_inactive()
