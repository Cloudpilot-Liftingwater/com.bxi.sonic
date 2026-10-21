"""Offline learned pose reference. Never feeds a decoded pose back to control."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from analysis_core import estimate_lag, metrics
from geometry import JOINTS
from eval_metrics import position_errors, summary
from execution_geometry import body_positions

ROOT=Path(__file__).resolve().parent

def unpack(raw):
    """Training concatenated ALL q frames then ALL dq frames before reshape."""
    raw=np.asarray(raw)
    if raw.ndim!=3 or raw.shape[1:]!=(10,64):raise ValueError('unexpected g1_kin shape')
    packed=raw[:,:,:58].reshape(len(raw),580)
    return packed[:,:290].reshape(-1,10,29),packed[:,290:].reshape(-1,10,29),raw[:,:,58:64]

class KinematicDecoder:
    def __init__(self):
        self.manifest=json.loads((ROOT/'kinematic_manifest.json').read_text())
        model=ROOT/'g1_kin_step28800.onnx'
        if hashlib.sha256(model.read_bytes()).hexdigest()!=self.manifest['kinematic_onnx_sha256']:
            raise ValueError('g1_kin model fingerprint differs from manifest')
        if tuple(self.manifest['joint_names'])!=JOINTS:raise ValueError('g1_kin joint order mismatch')
        import onnxruntime as ort
        options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
        self.session=ort.InferenceSession(str(model),sess_options=options,providers=['CPUExecutionProvider'])

    def decode(self,tokens):
        tokens=np.asarray(tokens,dtype=np.float32)
        if tokens.ndim!=2 or tokens.shape[1]!=64:raise ValueError('expected 64D recorded token')
        good=(np.isfinite(tokens).all(axis=1)&(tokens>=-1.00001).all(axis=1)&
              (tokens<=.93751).all(axis=1)&(np.abs(tokens*16-np.rint(tokens*16))<.0001).all(axis=1))
        raw=np.full((len(tokens),10,64),np.nan,dtype=np.float32)
        indices=np.flatnonzero(good)
        for start in range(0,len(indices),256):
            select=indices[start:start+256]
            raw[select]=self.session.run(None,{'motion_token':tokens[select].reshape(-1,2,32)})[0]
        return raw,good&np.isfinite(raw).all(axis=(1,2))

def associate(times,queries):
    """Recover v1's actual trace association. Ambiguous duplicate times refuse."""
    times=np.asarray(times);result=np.full(len(queries),-1,dtype=int)
    for k,t in enumerate(queries):
        if not np.isfinite(t):continue
        lo=np.searchsorted(times,t-1e-6);hi=np.searchsorted(times,t+1e-6,side='right')
        if hi-lo==1:result[k]=lo
    return result

def lag_segments(sessions,sequences):
    n=len(sessions)
    if not n:return np.zeros(0,dtype=int)
    return np.cumsum(np.r_[True,(sessions[1:]!=sessions[:-1])|(np.diff(sequences)!=1)])

def comparison_rows(times,reference,command,actual,valid,segments):
    result=[]
    for j,name in enumerate(JOINTS):
        result.append(dict(joint=name,
            reference_to_measured=metrics(np.where(valid,actual[:,j]-reference[:,j],np.nan)),
            command_to_measured=metrics(np.where(valid,actual[:,j]-command[:,j],np.nan)),
            reference_delay=estimate_lag(times,reference[:,j],actual[:,j],valid,epochs=segments),
            command_delay=estimate_lag(times,command[:,j],actual[:,j],valid,epochs=segments)))
    return result

def analyze_reference(out,original,times,degrees,valid,used_trace,arrays,geometry):
    manifest=json.loads((ROOT/'kinematic_manifest.json').read_text())
    models=original['quality']['unique_models']
    reference=np.full((len(times),29),np.nan);scored=np.zeros(len(times),bool)
    reason=None
    if len(models)!=1 or models[0]['sha256'] not in manifest['compatible_policy_sha256']:
        reason='记录中的模型 SHA 与已验证的 step28800 不匹配；不套用其他模型的参考。'
    elif original['quality']['counts'].get('ros_clock_discontinuities',0):
        reason='记录含 ROS 时钟跳变；本次不输出重建参考成绩，原始记录和执行报告保留。'
    elif tuple(arrays['joint_names'])!=JOINTS:reason='记录关节顺序与训练契约不匹配。'
    trace_q=np.full((len(arrays['monotonic']),10,29),np.nan)
    trace_dq=trace_q.copy();trace_rot=np.full((len(trace_q),10,6),np.nan)
    token_valid=np.zeros(len(trace_q),bool);index=associate(arrays['monotonic'],used_trace)
    segments=np.full(len(times),-1,dtype=int)
    if reason is None:
        decoder=KinematicDecoder()
        raw,token_valid=decoder.decode(arrays['motion_token'])
        trace_q,trace_dq,trace_rot=unpack(raw)
        trace_good=token_valid&arrays['source_live'].astype(bool)&~arrays['source_blend_active'].astype(bool)
        matched=index>=0
        reference[matched]=np.degrees(trace_q[index[matched],0])
        scored[matched]=(valid[matched]&trace_good[index[matched]]&np.isfinite(degrees[matched]).all(axis=(1,2)))
        epoch=lag_segments(arrays['session'],arrays['inference_sequence'])
        segments[matched]=epoch[index[matched]]
    rows=comparison_rows(times,reference,degrees[:,2],degrees[:,3],scored,segments)
    reference_bodies=body_positions(geometry,JOINTS,np.radians(reference))
    actual_bodies=body_positions(geometry,JOINTS,np.radians(degrees[:,3]))
    _,errors=position_errors(actual_bodies,reference_bodies);errors[~scored]=np.nan
    report=dict(status='unavailable' if reason else ('available' if scored.any() else 'no_valid_samples'),
        reason=reason,reference_kind=manifest['reference_kind'],is_independent_ground_truth=False,
        frames=int(scored.sum()),invalid_token_frames=int((~token_valid).sum()) if reason is None else None,
        joints=rows,reference_fixed_torso_body_error_mm=summary(errors,'mm'),
        coordinate_definition='same URDF, torso translation/rotation fixed on both sides; excludes whole-body heading/translation; not official ground-truth MPJPE_L',
        timing=dict(current_slot=0,robot_window_dt_s=manifest['robot_window_dt_s'],
            smpl_window_dt_s=manifest['smpl_window_dt_s'],
            alignment='state source time -> last command <= state time -> last completed inference <= command time; each step age <=50ms; exact same v1 association; current decoded reference held, never future window slot',
            reference_age_ms=summary((times-used_trace)[scored]*1000,'ms'),
            lag_definition='measured(t) vs held reference(t-lag), 0..250ms/5ms; zero-lag metrics always retained; no interpolation across gaps or session/sequence breaks; lag excludes upstream human capture time'),
        model=dict(policy_sha256=[m['sha256'] for m in models],
            checkpoint_sha256=manifest['checkpoint_sha256'],kinematic_onnx_sha256=manifest['kinematic_onnx_sha256']))
    np.savez_compressed(out/'kinematic_reference_arrays.npz',
        inference_monotonic=arrays['monotonic'],joint_names=np.array(JOINTS),
        reference_window_rad=trace_q,reference_velocity_window_rad_s=trace_dq,
        relative_anchor_rotation6d=trace_rot,token_valid=token_valid,
        reference_frame_index=arrays['reference_frame_index'],session=arrays['session'],
        current_slot=np.array(0),robot_window_dt_s=np.array(manifest['robot_window_dt_s']))
    with (out/'reference_joint_angles.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['monotonic_s','joint','reference_deg','decoder_deg','policy_deg','command_deg','measured_deg','scored','reference_age_ms','reference_frame_index'])
        for i,t in enumerate(times):
            for j,name in enumerate(JOINTS):
                values=[reference[i,j],*degrees[i,:,j]]
                w.writerow([t,name,*[float(v) if np.isfinite(v) else None for v in values],bool(scored[i]),
                    (t-used_trace[i])*1000 if np.isfinite(used_trace[i]) else None,
                    int(arrays['reference_frame_index'][index[i]]) if index[i]>=0 else None])
    with (out/'reference_joint_metrics.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['joint','comparison','n','mae_deg','rmse_deg','bias_deg','p95_abs_deg','max_abs_deg','estimated_lag_ms','lag_status','lag_compensated_rmse_deg','zero_lag_rmse_on_same_samples_deg'])
        for row in rows:
            for name,delay in [('reference_to_measured','reference_delay'),('command_to_measured','command_delay')]:
                m=row[name];d=row[delay]
                w.writerow([row['joint'],name,*[m[k] for k in ('n','mae_deg','rmse_deg','bias_deg','p95_abs_deg','max_abs_deg')],
                    d.get('estimated_lag_ms'),d['status'],d.get('lag_compensated_rmse_deg'),d.get('zero_lag_rmse_on_same_samples_deg')])
    return report,reference,scored,np.mean(errors,axis=1)
