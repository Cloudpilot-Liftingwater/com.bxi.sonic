import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from evaluate import BASE, evaluate
from kinematic_reference import KinematicDecoder, unpack, associate, comparison_rows, lag_segments, JOINTS
from reference_report import write_reference_report

def helper():
    spec=importlib.util.spec_from_file_location('tracking_fixture',BASE/'tests/test_tracking.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def fixture(tmp_path,seconds=1):
    h=helper();folder=h.synthetic_recording(tmp_path/'sample',seconds=seconds);h.analyze(folder)
    path=folder/'analysis/report.json';r=json.loads(path.read_text())
    r['quality']['unique_models'][0]['sha256']=json.loads((ROOT/'kinematic_manifest.json').read_text())['compatible_policy_sha256'][0]
    path.write_text(json.dumps(r));return folder

def test_real_export_matches_archived_pytorch_golden():
    decoder=KinematicDecoder()
    with np.load(ROOT/'tests/kinematic_golden.npz') as data:
        result,valid=decoder.decode(data['token'].reshape(-1,64))
        assert valid.all()
        np.testing.assert_allclose(result,data['raw'],atol=3e-5,rtol=3e-5)

def test_training_packing_restores_all_ten_frames_without_q_dq_mixing():
    q=np.arange(290).reshape(1,10,29)*.001
    dq=100+q;rot=np.arange(60).reshape(1,10,6)
    # Independently reproduce TrackingCommand.cat(q_flat,dq_flat), then reshape.
    raw=np.concatenate([np.concatenate([q.reshape(1,-1),dq.reshape(1,-1)],axis=-1).reshape(1,10,58),rot],axis=-1)
    got_q,got_dq,got_rot=unpack(raw)
    np.testing.assert_array_equal(got_q,q);np.testing.assert_array_equal(got_dq,dq);np.testing.assert_array_equal(got_rot,rot)
    assert got_q[0,1,0]!=raw[0,1,0]  # catches the tempting but incorrect [:,:,:29]

def test_invalid_tokens_are_missing_not_silently_requantized():
    token=np.zeros((4,64),np.float32);token[1,2]=np.nan;token[2,1]=.01;token[3,4]=1.
    raw,valid=KinematicDecoder().decode(token)
    assert valid.tolist()==[True,False,False,False]
    assert np.isnan(raw[1:]).all()

def test_trace_association_rejects_duplicates_and_distant_samples():
    np.testing.assert_array_equal(associate([1.,2.,2.,3.],[1.+1e-8,2.,2.9,np.nan]),[0,-1,-1,-1])

def test_session_restart_and_sequence_gap_break_lag_segments():
    result=lag_segments(np.array(['a','a','a','b','b']),np.array([1,2,4,1,2]))
    assert result.tolist()==[1,1,2,3,3]

def test_two_errors_known_lag_and_no_compensation_hiding():
    t=np.arange(0,6,.005);ref=20*np.sin(4*t)
    command=ref+3;actual=20*np.sin(4*(t-.05))+2
    expand=lambda x:np.repeat(x[:,None],29,axis=1)
    rows=comparison_rows(t,expand(ref),expand(command),expand(actual),np.ones(len(t),bool),np.zeros(len(t)))
    r=rows[0];d=r['reference_delay']
    assert abs(d['estimated_lag_ms']-50)<=5
    assert d['lag_compensated_rmse_deg']==pytest.approx(2,abs=.15)
    assert r['reference_to_measured']['rmse_deg']>d['lag_compensated_rmse_deg']
    assert r['command_to_measured']['rmse_deg']!=r['reference_to_measured']['rmse_deg']
    assert d['zero_lag_rmse_on_same_samples_deg']>d['lag_compensated_rmse_deg']

def test_end_to_end_real_decoder_and_four_curves(tmp_path):
    folder=fixture(tmp_path);report=evaluate(folder);r=report['kinematic_reference_tracking']
    assert r['status']=='available' and r['frames']>0
    assert r['timing']['current_slot']==0
    assert r['timing']['robot_window_dt_s']==.1 and r['timing']['smpl_window_dt_s']==.02
    assert not r['is_independent_ground_truth']
    assert all(row['reference_to_measured']['n']==r['frames'] for row in r['joints'])
    page=(folder/'analysis/tracking_reference.html').read_text()
    assert '重建参考 g1_kin' in page and '原始 RMSE' in page and '合成测试数据' in page
    payload=json.loads(page.split('<script type="application/json" id="data">')[1].split('</script>')[0])
    assert len(payload['q'])==4
    with np.load(folder/'analysis/kinematic_reference_arrays.npz') as data:
        assert data['reference_window_rad'].shape[1:]==(10,29)
    assert (folder/'analysis/reference_joint_metrics.csv').exists()

def test_only_current_slot_is_scored(tmp_path,monkeypatch):
    folder=fixture(tmp_path)
    def output(self,tokens):
        n=len(tokens);q=np.zeros((n,10,29));q[:,1:]=12345
        packed=np.concatenate([q.reshape(n,290),np.zeros((n,290))],axis=1).reshape(n,10,58)
        return np.concatenate([packed,np.zeros((n,10,6))],axis=2),np.ones(n,bool)
    monkeypatch.setattr(KinematicDecoder,'decode',output)
    r=evaluate(folder)['kinematic_reference_tracking']
    assert r['status']=='available'
    assert max(row['reference_to_measured']['max_abs_deg'] for row in r['joints'])<100

@pytest.mark.parametrize('change',['wrong_model','clock_jump','no_model'])
def test_unverified_model_or_clock_never_produces_fake_reference(tmp_path,change):
    folder=fixture(tmp_path);path=folder/'analysis/report.json';r=json.loads(path.read_text())
    if change=='wrong_model':r['quality']['unique_models'][0]['sha256']='f'*64
    elif change=='no_model':r['quality']['unique_models']=[]
    else:r['quality']['counts']['ros_clock_discontinuities']=1
    path.write_text(json.dumps(r));result=evaluate(folder)['kinematic_reference_tracking']
    assert result['status']=='unavailable' and result['frames']==0
    assert result['reference_fixed_torso_body_error_mm']['mean'] is None
    assert all(row['reference_to_measured']['rmse_deg'] is None for row in result['joints'])

def test_graph_decimation_cannot_bridge_invalid_run(tmp_path):
    n=6001;t=np.arange(n)*.005;valid=np.ones(n,bool);valid[3]=False
    report=dict(kinematic_reference_tracking={})
    path=tmp_path/'report.html';write_reference_report(path,report,t,np.zeros((n,29)),np.zeros((n,4,29)),valid)
    payload=json.loads(path.read_text().split('<script type="application/json" id="data">')[1].split('</script>')[0])
    assert payload['stride']==2 and payload['valid'][2] is False
