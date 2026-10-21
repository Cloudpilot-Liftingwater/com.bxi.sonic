"""Offline, camera-free SONIC tracking audit. No ROS or robot connection needed."""
import argparse
import base64
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

from analysis_core import causal_indices, estimate_lag, metrics, named, packed, source_time
from geometry import BENDS, CONTRACT, JOINTS, SMPL_BODY, RobotGeometry, human_bends
from tracking_tap import decode

WRISTS = (19,20,21,26,27,28)


def signature(points, root, wrist):
    return hashlib.sha256(b''.join(np.asarray(a,dtype='<f4').tobytes() for a in (points,root,wrist))).digest()


def clean(value):
    if isinstance(value,dict):return {k:clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [clean(v) for v in value]
    if isinstance(value,np.ndarray):return clean(value.tolist())
    if isinstance(value,(float,np.floating)) and not np.isfinite(value):return None
    if isinstance(value,np.generic):return value.item()
    return value


def dump(path,value):
    path.write_text(json.dumps(clean(value),ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def analyze(folder):
    folder=Path(folder)
    out=folder/'analysis';out.mkdir(exist_ok=True)
    counters=Counter();commands=[];states=[];traces=[];modes=[]
    human_lookup=defaultdict(list);seen_human={};human_epoch=0;last_frame=-1
    epoch=0;clock_offset=None
    human_file=(out/'human_smpl_angles.csv').open('w',newline='')
    human_writer=csv.writer(human_file)
    human_writer.writerow(['pico_epoch','frame_index','first_received_monotonic_s']+
        [f'{n}.rotvec_{axis}_deg' for n in SMPL_BODY for axis in 'xyz'])
    try:
        with (folder/'events.jsonl').open() as stream:
            for line in stream:
                try:
                    e=json.loads(line);kind=e['kind'];value=e['value'];counters[kind+'_events']+=1
                    offset=e['ros_now_ns']*1e-9-e['received_monotonic']
                    if clock_offset is not None and abs(offset-clock_offset)>.02:
                        epoch+=1;counters['ros_clock_discontinuities']+=1
                    clock_offset=offset
                    if kind in ('command','state'):
                        t=source_time(e)
                        q=named(value['position'],value['names'],JOINTS)
                        row=dict(t=t,epoch=epoch,q=q)
                        if kind=='command':
                            row['kp']=named(value['kp'],value['names'],JOINTS)
                            commands.append(row)
                        else:states.append(row)
                    elif kind=='state_machine':
                        modes.append((e['received_monotonic'],value.get('current',{}).get('name',''),epoch))
                    elif kind=='trace':
                        meta,arrays=decode(base64.b64decode(value['packet_b64'],validate=True))
                        if tuple(meta['joint_names'])!=JOINTS:raise ValueError('trace joint layout differs from reviewed contract')
                        if len(meta['model_sha256'])!=64:raise ValueError('missing model fingerprint')
                        traces.append(dict(t=meta['completed_monotonic'],epoch=epoch,meta=meta,**arrays))
                    elif kind=='pose':
                        fields=packed(base64.b64decode(value['packet_b64'],validate=True),'pose')
                        ids=np.asarray(fields['frame_index']).reshape(-1)
                        poses=np.asarray(fields['smpl_pose']).reshape(len(ids),21,3)
                        points=np.asarray(fields['smpl_joints']).reshape(len(ids),24,3)
                        roots=np.asarray(fields['body_quat_w']).reshape(len(ids),4)
                        wrists=np.asarray(fields['joint_pos']).reshape(len(ids),29)[:,WRISTS]
                        if not all(np.isfinite(a).all() for a in (poses,points,roots,wrists)):
                            raise ValueError('nonfinite PICO pose')
                        if len(ids) and int(ids[-1])<last_frame:human_epoch+=1
                        if len(ids):last_frame=int(ids[-1])
                        for i,index in enumerate(ids):
                            identity=(human_epoch,int(index));rot=poses[i].astype('<f4')
                            h=hashlib.sha256(rot.tobytes()).hexdigest()
                            if identity in seen_human:
                                if seen_human[identity]!=h:counters['pico_frame_content_conflicts']+=1
                                continue
                            seen_human[identity]=h
                            human_writer.writerow([human_epoch,int(index),e['received_monotonic']]+np.degrees(rot).reshape(-1).tolist())
                            human_lookup[signature(points[i],roots[i],wrists[i])].append(
                                (e['received_monotonic'],rot.copy(),identity))
                except (ValueError,KeyError,TypeError,IndexError,OverflowError) as exc:
                    counters['rejected: '+str(exc)]+=1
    finally:human_file.close()
    commands.sort(key=lambda x:x['t']);states.sort(key=lambda x:x['t']);traces.sort(key=lambda x:x['t']);modes.sort()
    # Sequence gaps/restarts are preserved and quantified, never silently filled.
    last={}
    for tr in traces:
        m=tr['meta'];session=m['session'];sequence=int(m['sequence'])
        if session in last:
            counters['trace_sequence_gaps']+=max(0,sequence-last[session]-1)
            counters['trace_repeated_or_reversed_sequence']+=int(sequence<=last[session])
        last[session]=sequence
    ct=np.array([x['t'] for x in commands]);st=np.array([x['t'] for x in states]);tt=np.array([x['t'] for x in traces])
    n=len(states);actual=np.array([x['q'] for x in states]).reshape(n,29)
    command=np.full((n,29),np.nan);policy=command.copy();decoder=command.copy();reference_wrist=command.copy()
    kp=command.copy();command_age=np.full(n,np.nan);trace_age=np.full(n,np.nan)
    valid_live=np.zeros(n,dtype=bool);matched_trace=np.full(n,-1,dtype=int)
    ci=causal_indices(ct,st,max_age=.05);ti=causal_indices(tt,ct,max_age=.05)
    mode_i=causal_indices([m[0] for m in modes],st,max_age=.5)
    models={(x['meta']['model_id'],x['meta']['model_sha256']) for x in traces}
    for i,c in enumerate(ci):
        if c<0 or states[i]['epoch']!=commands[c]['epoch']:continue
        command[i]=commands[c]['q'];kp[i]=commands[c]['kp'];command_age[i]=st[i]-ct[c]
        t=ti[c]
        if t<0 or traces[t]['epoch']!=states[i]['epoch']:continue
        tr=traces[t];meta=tr['meta'];matched_trace[i]=t;trace_age[i]=ct[c]-tt[t]
        policy[i]=tr['policy_target'];decoder[i]=tr['decoded_target']
        reference_wrist[i,list(WRISTS)]=tr['model_input'][780:786]
        m=mode_i[i]
        valid_live[i]=(meta['source']=='live' and meta['policy_status']=='live_reference'
            and not meta['source_blend_active'] and len(models)==1 and m>=0
            and modes[m][1]=='com.bxi.sonic/sonic_teleop' and modes[m][2]==states[i]['epoch'])
    valid=np.isfinite(actual)&np.isfinite(command)&np.isfinite(kp)&(kp>0)&valid_live[:,None]
    q=[np.degrees(a) for a in (decoder,policy,command,actual,reference_wrist)]
    error=np.where(valid,q[3]-q[2],np.nan)
    robot_rows=[]
    for j,name in enumerate(JOINTS):
        robot_rows.append(dict(joint=name,**metrics(error[:,j]),
            valid_fraction_of_received_states=float(valid[:,j].mean()) if n else 0.,
            policy_to_final_command=metrics(np.where(valid[:,j],q[2][:,j]-q[1][:,j],np.nan)),
            decoder_to_measured=metrics(np.where(valid[:,j],q[3][:,j]-q[0][:,j],np.nan)),
            delay=estimate_lag(st,q[2][:,j],q[3][:,j],valid[:,j],epochs=[x['epoch'] for x in states])))
    consumed_pose=np.full((len(traces),10,21,3),np.nan,dtype=np.float32)
    pose_matches=Counter()
    human_shape=np.full((len(traces),4),np.nan)
    decoded_shape=human_shape.copy();policy_shape=human_shape.copy()
    geometry=RobotGeometry()
    for i,tr in enumerate(traces):
        encoder=tr['model_input'][:840];points=encoder[:720].reshape(10,24,3)
        roots=tr['root_quat'].reshape(10,4);wrists=encoder[780:840].reshape(10,6)
        human_shape[i]=human_bends(points[0])
        decoded_shape[i]=geometry.bends(tr['decoded_target'])
        policy_shape[i]=geometry.bends(tr['policy_target'])
        for k in range(10):
            candidates=[x for x in human_lookup.get(signature(points[k],roots[k],wrists[k]),())
                        if -2. <= x[0]-tr['t'] <= .1]
            distinct={x[1].tobytes():x[1] for x in candidates}
            if len(distinct)==1:
                consumed_pose[i,k]=next(iter(distinct.values()));pose_matches['unique_exact_tensor_match']+=1
            elif distinct:pose_matches['ambiguous_twist_angles']+=1
            else:pose_matches['no_exact_source_match']+=1
    human=np.full((n,4),np.nan);body_decoder=human.copy();body_policy=human.copy();body_command=human.copy();body_actual=human.copy()
    for i,t in enumerate(matched_trace):
        if t<0:continue
        human[i]=human_shape[t];body_decoder[i]=decoded_shape[t];body_policy[i]=policy_shape[t]
        body_command[i]=geometry.bends(command[i]);body_actual[i]=geometry.bends(actual[i])
    shape_valid=valid.all(axis=1)
    body_rows=[dict(joint=name,
        human_to_decoder=metrics(np.where(shape_valid,body_decoder[:,j]-human[:,j],np.nan)),
        human_to_measured=metrics(np.where(shape_valid,body_actual[:,j]-human[:,j],np.nan)),
        command_to_measured=metrics(np.where(shape_valid,body_actual[:,j]-body_command[:,j],np.nan))) for j,name in enumerate(BENDS)]
    capture=json.loads((folder/'capture_result.json').read_text()) if (folder/'capture_result.json').exists() else dict(missing=['capture_result'])
    valid_count=int(valid.any(axis=1).sum())
    limitations=[
        '主误差=同一ROS源时刻的实际关节角−最近已发布目标角；目标保持，最大年龄50ms，不用未来指令配对。',
        '只统计新鲜SONIC live、无源切换混合、KP>0且时间/关节有效的样本；全部原始记录保留。',
        'SMPL旋转向量分量不是机器人单轴关节角。只对六个原生腕角和肘膝几何弯曲角作有定义的参考比较。',
        '完整encoder窗口在输入中，不把广播收到的最新窗口冒充实际被模型消费的窗口。',
        '人体旋转角与输入位置逐帧按位置+root+wrist内容精确匹配；有歧义/缺失时保留NaN，不能由位置反推关节扭转角。',
        '关节指令是ROS最终发布值，不是电机驱动接收ACK；反馈是驱动发布的关节坐标，不是外部动捕真值或原始转子角。',
        '弯曲角使用随包URDF，尚未与机器人当前URDF指纹核对；为姿态形状诊断，不是全身重定向精度认证。',
        '延迟为运动曲线估计，静止/低激励不给结论；原始零延迟误差始终保留。']
    metadata=json.loads((folder/'metadata.json').read_text()) if (folder/'metadata.json').exists() else {}
    def timing_summary(t):
        if len(t)<2:return dict(samples=len(t),estimated_hz=None)
        dt=np.diff(t)
        return dict(samples=len(t),estimated_hz=float((len(t)-1)/(t[-1]-t[0])) if t[-1]>t[0] else None,
            median_interval_ms=float(np.median(dt)*1000),p95_interval_ms=float(np.percentile(dt,95)*1000),
            max_interval_ms=float(dt.max()*1000),nonprogressing=int((dt<=0).sum()))
    quality=dict(capture=capture,counts=dict(counters),unique_models=[dict(id=a,sha256=b) for a,b in sorted(models)],
        received_state_rows=n,live_state_rows=int(valid_live.sum()),scored_state_rows=valid_count,
        unmatched_commands=int((ci<0).sum()),unmatched_trace=int((matched_trace<0).sum()),
        human_encoder_matching=dict(pose_matches),trace_sessions=len(last),
        max_observed_tap_overhead_us=max((x['meta'].get('tap_previous_max_overhead_ns',0) for x in traces),default=0)/1000,
        tap_drop_counter_max=max((x['meta'].get('tap_dropped',0) for x in traces),default=0),
        tap_error_counter_max=max((x['meta'].get('tap_errors',0) for x in traces),default=0),
        source_blended_inferences=sum(bool(x['meta']['source_blend_active']) for x in traces),
        action_clipped_components=sum(int(np.count_nonzero(x['raw_action']!=x['clipped_action'])) for x in traces),
        invalid_state_components=int((~np.isfinite(actual)).sum()),
        invalid_command_components=sum(int((~np.isfinite(x['q'])).sum()) for x in commands),
        invalid_gain_components=sum(int((~np.isfinite(x['kp'])).sum()) for x in commands),
        stream_timing=dict(state=timing_summary(st),command=timing_summary(ct),trace=timing_summary(tt)),
        reference_urdf_sha256=CONTRACT['reference_urdf_sha256'],reference_urdf_remote_verified=False)
    complete=(valid_count>0 and not capture.get('missing') and not capture.get('writer_errors')
        and not sum(capture.get('queue_lost',{}).values()) and not capture.get('errors')
        and not counters['ros_clock_discontinuities'] and not counters['trace_sequence_gaps']
        and not any(v for k,v in counters.items() if k.startswith('rejected:'))
        and not quality['invalid_state_components'] and not quality['invalid_command_components']
        and not quality['invalid_gain_components'] and not counters['pico_frame_content_conflicts']
        and not pose_matches['no_exact_source_match'] and not pose_matches['ambiguous_twist_angles'])
    report=dict(status='OBSERVED_DATA_COMPLETE' if complete else 'PARTIAL_OR_NO_VALID_SAMPLES',
        synthetic=bool(metadata.get('synthetic',False)),
        units='degrees',error_sign='measured minus target',quality=quality,
        joints=robot_rows,body_shape=body_rows,limitations=limitations)
    dump(out/'report.json',report)
    with (out/'joint_metrics.csv').open('w',newline='') as f:
        columns=['joint','n','mae_deg','rmse_deg','bias_deg','p95_abs_deg','max_abs_deg','valid_fraction_of_received_states']
        w=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore');w.writeheader();w.writerows(robot_rows)
    with (out/'aligned_joint_angles.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['monotonic_s','joint','decoder_deg','policy_deg','command_deg','measured_deg',
            'encoder_wrist_deg','error_deg','valid','command_age_ms','trace_to_command_ms'])
        for i in range(n):
            for j,name in enumerate(JOINTS):
                w.writerow(clean([st[i],name,*[a[i,j] for a in q],error[i,j],bool(valid[i,j]),command_age[i]*1000,trace_age[i]*1000]))
    with (out/'body_shape_angles.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['monotonic_s','joint','human_bend_deg','decoder_bend_deg','policy_bend_deg','command_bend_deg','measured_bend_deg','live'])
        for i in range(n):
            for j,name in enumerate(BENDS):w.writerow(clean([st[i],name,human[i,j],body_decoder[i,j],body_policy[i,j],body_command[i,j],body_actual[i,j],bool(shape_valid[i])]))
    np.savez_compressed(out/'inference_arrays.npz',
        monotonic=tt,joint_names=np.array(JOINTS),
        model_input=np.array([x['model_input'] for x in traces],dtype=np.float32).reshape(-1,1770),
        raw_action=np.array([x['raw_action'] for x in traces],dtype=np.float32).reshape(-1,29),
        clipped_action=np.array([x['clipped_action'] for x in traces],dtype=np.float32).reshape(-1,29),
        decoded_target_rad=np.array([x['decoded_target'] for x in traces],dtype=np.float32).reshape(-1,29),
        policy_target_rad=np.array([x['policy_target'] for x in traces],dtype=np.float32).reshape(-1,29),
        observed_q_used_rad=np.array([x['observed_q'] for x in traces],dtype=np.float32).reshape(-1,29),
        motion_token=np.array([x['motion_token'] for x in traces],dtype=np.float32).reshape(-1,64),
        consumed_smpl_rotvec_rad=consumed_pose,
        session=np.array([x['meta']['session'] for x in traces]),
        inference_sequence=np.array([x['meta']['sequence'] for x in traces],dtype=np.int64),
        source_live=np.array([x['meta']['source']=='live' for x in traces]),
        source_blend_active=np.array([x['meta']['source_blend_active'] for x in traces]),
        reference_frame_index=np.array([x['meta']['reference_frame_index'] for x in traces],dtype=np.int64))
    from html_report import write_html
    write_html(out/'report.html',report,st,q,valid,human,body_decoder,body_policy,body_command,body_actual,shape_valid)
    print('ANALYSIS_DIR='+str(out.resolve()))
    print('TRACKING_RESULT='+json.dumps(dict(status=report['status'],received=n,scored=valid_count),ensure_ascii=False))
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('recording',type=Path)
    a=p.parse_args();analyze(a.recording)
