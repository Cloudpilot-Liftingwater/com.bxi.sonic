import csv
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from eval_metrics import BODIES, position_errors, quaternion_error_degrees, sixd_error_degrees, world_metrics
from evaluate import BASE, JOINTS, CONTRACT, RobotGeometry, evaluate, matched_trace_mask
from execution_geometry import body_positions


def test_matches_actual_smplsim_function_golden():
    with np.load(Path(__file__).with_name('official_golden.npz'),allow_pickle=False) as d:
        g,l=position_errors(d['pred'],d['ref'])
        np.testing.assert_allclose(g,d['global_mm'],atol=1e-10)
        np.testing.assert_allclose(l,d['local_mm'],atol=1e-10)
        np.testing.assert_allclose(quaternion_error_degrees(d['pred_quat'],d['ref_quat']),d['rotation_degrees'],atol=1e-10)


def test_global_translation_is_removed_only_from_local():
    ref=np.zeros((2,3,3));ref[:,1,0]=1;ref[:,2,1]=1
    g,l=position_errors(ref+[1,0,0],ref)
    np.testing.assert_allclose(g,1000);np.testing.assert_allclose(l,0)


def test_local_metric_retains_heading_error():
    ref=np.array([[[0.,0,0],[1,0,0],[0,1,0]]])
    rot=np.array([[0.,-1,0],[1,0,0],[0,0,1]])
    g,l=position_errors(ref@rot.T,ref)
    np.testing.assert_allclose(g,l)
    assert l.mean()>900


def test_relative_rotation_uses_policy_row_major_sixd():
    # Policy flattening is [R00,R01,R10,R11,R20,R21].
    a=np.deg2rad(37);r=np.array([[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1]])
    assert sixd_error_degrees(r[:,:2].reshape(1,6))[0]==pytest.approx(37)
    assert np.isnan(sixd_error_degrees(np.zeros((1,6)))[0])


def test_quaternion_sign_and_invalid_zero():
    result=quaternion_error_degrees([[0,0,0,-1],[0,0,0,0]],[[0,0,0,1],[0,0,0,1]])
    assert result[0]==0 and np.isnan(result[1])


def test_fk_matches_independent_urdf_joint_formula():
    g=RobotGeometry();q=np.zeros((2,29));q[1,JOINTS.index('l_elbow_y_joint')]=.5
    p=body_positions(g,JOINTS,q)
    wrist=BODIES.index('l_wrist_z_link');elbow=BODIES.index('l_elbow_y_link')
    assert np.linalg.norm(p[1,wrist]-p[0,wrist])>.05
    np.testing.assert_allclose(p[1,elbow],p[0,elbow])
    np.testing.assert_allclose(p[1,BODIES.index('r_wrist_z_link')],p[0,BODIES.index('r_wrist_z_link')])
    np.testing.assert_allclose(p[:,0],0)
    # Known first lower-body body derives from explicit chained URDF transforms.
    links={'torso_link':np.eye(4)}
    for name,kind,parent,child,origin,axis in g.joints:
        if parent in links:links[child]=links[parent]@origin
    np.testing.assert_allclose(p[0,BODIES.index('l_hip_x_link')],links['l_hip_x_link'][:3,3])


def world_file(path,**overrides):
    p=np.zeros((3,len(BODIES),3));p[:,:,0]=np.arange(len(BODIES))*.1
    fields=dict(time_s=np.arange(3)*.02,predicted_world_m=p+[.1,0,0],reference_world_m=p,
        body_names=np.array(BODIES),coordinate_frame='lab_world',reference_kind='retargeted_robot',
        actual_kind='external_mocap',synchronization='synthetic same timestamps',
        robot_model_sha256='a'*64,reference_robot_model_sha256='a'*64)
    fields.update(overrides);np.savez_compressed(path,**fields);return path


def test_world_contract_computes_global_local_and_root(tmp_path):
    r=world_metrics(world_file(tmp_path/'world.npz'))
    assert r['mpjpe_g_mm']['mean']==pytest.approx(100)
    assert r['mpjpe_l_mm']['mean']==pytest.approx(0,abs=1e-10)
    assert r['root_position_error_mm']['mean']==pytest.approx(100)
    assert r['root_orientation_error_deg']['mean'] is None


def test_world_root_rotation_reports_degrees(tmp_path):
    q=np.tile([0.,0.,np.sqrt(.5),np.sqrt(.5)],(3,1))
    r=world_metrics(world_file(tmp_path/'world.npz',predicted_root_xyzw=q,
        reference_root_xyzw=np.tile([0.,0.,0.,1.],(3,1))))
    assert r['root_orientation_error_deg']['mean']==pytest.approx(90)


@pytest.mark.parametrize('change',[{'reference_kind':'decoder'}, {'reference_robot_model_sha256':'b'*64},
    {'body_names':np.array(BODIES[::-1])}, {'coordinate_frame':''}, {'time_s':[0.,0.,.1]}])
def test_world_contract_rejects_fake_or_unaligned_sources(tmp_path,change):
    with pytest.raises(ValueError):world_metrics(world_file(tmp_path/'world.npz',**change))


def test_missing_world_values_do_not_become_zero(tmp_path):
    p=np.zeros((3,len(BODIES),3));p[1]=np.nan
    r=world_metrics(world_file(tmp_path/'world.npz',predicted_world_m=p))
    assert r['valid_frames']==2


def test_trace_association_does_not_accept_distant_time():
    np.testing.assert_array_equal(matched_trace_mask([1.,2.],[1.+1e-8,1.98],[True,True]),[True,False])


def test_real_v1_outputs_accepted_without_reinstall(tmp_path):
    spec=importlib.util.spec_from_file_location('v1_test_helpers',BASE/'tests/test_tracking.py')
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    folder=helper.synthetic_recording(tmp_path/'sample',seconds=.5)
    helper.analyze(folder)
    r=evaluate(folder)
    assert r['global_reference_tracking']['status']=='unavailable'
    assert r['global_reference_tracking']['mpjpe_g_mm'] is None
    assert not r['execution_shape']['is_official_reference_tracking_mpjpe']
    assert r['execution_shape']['comparisons'][0]['root_fixed_body_error_mm']['mean']==pytest.approx(0)
    assert r['execution_shape']['comparisons'][2]['root_fixed_body_error_mm']['mean']>0
    assert r['encoder_anchor_orientation']['n']==0  # v1 fixture deliberately has invalid zero 6D
    assert (folder/'analysis/eval_tracking.png').exists()
    assert '合成测试数据' in (folder/'analysis/eval_report.html').read_text()
    path=folder/'analysis/inference_arrays.npz'
    with np.load(path,allow_pickle=False) as data:arrays={k:data[k].copy() for k in data.files}
    arrays['model_input'][:,720:726]=[0.,-1.,1.,0.,0.,0.]
    np.savez_compressed(path,**arrays)
    updated=evaluate(folder)
    assert updated['encoder_anchor_orientation']['n']==30  # first 20 inference frames are blended
    assert updated['encoder_anchor_orientation']['mean']==pytest.approx(90)


def test_empty_recording_is_not_zero_error(tmp_path):
    from analyze import analyze
    (tmp_path/'events.jsonl').write_text('')
    analyze(tmp_path);r=evaluate(tmp_path)
    assert r['encoder_anchor_orientation']['mean'] is None
    assert all(x['root_fixed_body_error_mm']['mean'] is None for x in r['execution_shape']['comparisons'])
