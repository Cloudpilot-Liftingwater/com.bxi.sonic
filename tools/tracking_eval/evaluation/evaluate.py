"""Add official-definition diagnostics to an existing SONIC tracking recording.

Offline only. Reads v1 analysis outputs; never imports ROS or changes the policy.
"""
import argparse
import csv
import hashlib
import html
from itertools import groupby
import json
import os
from pathlib import Path
import sys
import numpy as np

BASE=Path(__file__).resolve().parent.parent/'recording'
sys.path.insert(0,str(BASE))
from geometry import JOINTS, CONTRACT, RobotGeometry
from eval_metrics import BODIES, position_errors, sixd_error_degrees, summary, world_metrics
from execution_geometry import body_positions
from kinematic_reference import analyze_reference
from reference_report import write_reference_report

STAGES=('decoder','policy','command','measured')
PAIRS=((0,1,'decoder_to_policy'),(1,2,'policy_to_command'),(2,3,'command_to_measured'),(0,3,'decoder_to_measured'))


def read_aligned(path):
    times=[];angles=[];valid=[];trace_times=[]
    with path.open() as f:
        reader=csv.DictReader(f)
        for t,group in groupby(reader,key=lambda row:float(row['monotonic_s'])):
            rows=list(group)
            if tuple(row['joint'] for row in rows)!=JOINTS:raise ValueError('aligned CSV joint layout differs')
            times.append(t)
            angles.append([[float(row[stage+'_deg']) if row[stage+'_deg'] else np.nan for row in rows] for stage in STAGES])
            valid.append(all(row['valid']=='True' for row in rows))
            first=rows[0]
            trace_times.append(t-(float(first['command_age_ms'])+float(first['trace_to_command_ms']))*.001
                if first['command_age_ms'] and first['trace_to_command_ms'] else np.nan)
    return np.asarray(times),np.asarray(angles).reshape(-1,4,29),np.asarray(valid,dtype=bool),np.asarray(trace_times)


def matched_trace_mask(times, queries, valid):
    times=np.asarray(times);mask=np.zeros(len(times),dtype=bool)
    if not len(times):return mask
    for query in np.asarray(queries)[np.asarray(valid)&np.isfinite(queries)]:
        i=int(np.searchsorted(times,query))
        candidates=[j for j in (i-1,i) if 0<=j<len(times)]
        j=min(candidates,key=lambda x:abs(times[x]-query))
        if abs(times[j]-query)<1e-6:mask[j]=True
    return mask


def evaluate(recording, paired_world=None):
    folder=Path(recording);out=folder/'analysis'
    original=json.loads((out/'report.json').read_text())
    times,degrees,valid,used_trace=read_aligned(out/'aligned_joint_angles.csv')
    with np.load(out/'inference_arrays.npz',allow_pickle=False) as data:
        arrays={k:data[k].copy() for k in data.files}
        trace_t=arrays['monotonic'].copy()
        relative6d=arrays['model_input'][:,720:726]
        ori=sixd_error_degrees(relative6d)
        trace_valid=matched_trace_mask(trace_t,used_trace,valid)&arrays['source_live'].astype(bool)&~arrays['source_blend_active'].astype(bool)
    model=BASE/'elf3_reference.urdf'
    if hashlib.sha256(model.read_bytes()).hexdigest()!=CONTRACT['reference_urdf_sha256']:
        raise ValueError('reference URDF fingerprint changed; review before using FK metrics')
    geometry=RobotGeometry(model)
    bodies=[body_positions(geometry,JOINTS,np.radians(degrees[:,i])) for i in range(4)]
    rows=[];curves={}
    for a,b,name in PAIRS:
        _,err=position_errors(bodies[b],bodies[a])
        err[~valid]=np.nan
        curves[name]=np.mean(err,axis=1)
        rows.append(dict(comparison=name,frames=int(np.isfinite(err).all(axis=1).sum()),
            root_fixed_body_error_mm=summary(err,'mm'),
            per_body=[dict(body=k,**summary(err[:,j],'mm')) for j,k in enumerate(BODIES)]))
    global_result=dict(status='unavailable',mpjpe_g_mm=None,mpjpe_l_mm=None,
        root_position_error_mm=None,root_orientation_error_deg=None,
        reason='No independently retargeted full ELF3 reference trajectory and synchronized robot world pose in the v1 live stream. SMPL local positions and decoder targets cannot replace these sources.')
    if paired_world is not None:global_result=world_metrics(paired_world)
    reference_result,reference,reference_valid,reference_curve=analyze_reference(
        out,original,times,degrees,valid,used_trace,arrays,geometry)
    curves['kinematic_reference_to_measured']=reference_curve
    report=dict(release='sonic_eval_20261015_v2',synthetic=original.get('synthetic',False),
        recording_status=original['status'],
        kinematic_reference_tracking=reference_result,
        global_reference_tracking=global_result,
        encoder_anchor_orientation=dict(**summary(ori[trace_valid],'deg'),
            valid_live_inferences=int(trace_valid.sum()),invalid_rotation_frames=int((trace_valid&~np.isfinite(ori)).sum()),
            definition='SO(3) angle between consumed SMPL root orientation and robot waist_z anchor, after policy yaw_offset alignment; slot 0, observation at inference input.',
            is_absolute_world_orientation_error=False),
        execution_shape=dict(frame='torso fixed to identity on both sides',body_names=BODIES,local_root='torso_link',
            comparisons=rows,urdf_sha256=CONTRACT['reference_urdf_sha256'],urdf_remote_verified=False,
            is_official_reference_tracking_mpjpe=False,
            limitation='Same robot FK compares targets against measured joints. It excludes base translation/rotation and cannot prove human-to-robot imitation accuracy.'),
        official_definition=dict(mpjpe_g='mean Euclidean world body-position error, millimetres',
            mpjpe_l='subtract each trajectory root position, then mean Euclidean error; no rotation alignment',
            local_elf3_root='torso_link (body index 0), not the waist_z anchor',
            rotation='Optional SMPLSim rot_error uses geodesic radians; current official default callback does not pass robot rotations. This diagnostic displays degrees.',
            success_rate='Unavailable for free teleoperation: no defined evaluation motions or identical termination criteria.'))
    (out/'eval_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    with (out/'execution_shape_errors.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['monotonic_s','valid']+list(curves))
        for i,t in enumerate(times):w.writerow([t,bool(valid[i])]+[None if not np.isfinite(v[i]) else v[i] for v in curves.values()])
    with (out/'encoder_anchor_orientation.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['inference_monotonic_s','scored','yaw_aligned_anchor_error_deg'])
        for t,v,ok in zip(trace_t,ori,trace_valid):w.writerow([t,bool(ok),float(v) if np.isfinite(v) else None])
    render(out,report,times,curves,trace_t,np.where(trace_valid,ori,np.nan))
    write_reference_report(out/'tracking_reference.html',report,times,reference,degrees,reference_valid)
    print('EVAL_REPORT='+str((out/'eval_report.html').resolve()))
    print('WORLD_REFERENCE_METRICS='+global_result['status'])
    print('KINEMATIC_REFERENCE='+reference_result['status'])
    print('TRACKING_REFERENCE_REPORT='+str((out/'tracking_reference.html').resolve()))
    print('HTML_REPORT='+str((out/'tracking_reference.html').resolve()))
    return report


def render(out,report,times,curves,trace_t,ori):
    os.environ.setdefault('MPLCONFIGDIR',str(out.resolve()/'.matplotlib_cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    origin=times[0] if len(times) else (trace_t[0] if len(trace_t) else 0.)
    fig,axes=plt.subplots(2,1,figsize=(12,7),layout='constrained')
    def break_gaps(t,values):
        y=np.asarray(values).copy()
        if len(t)>1:y[np.r_[False,(np.diff(t)>.06)|(np.diff(t)<=0)]]=np.nan
        return y
    for name,values in curves.items():axes[0].plot(times-origin,break_gaps(times,values),label=name,linewidth=1.)
    axes[0].set(title='Execution shape: fixed-torso FK (not global reference MPJPE)',ylabel='Mean body position error (mm)')
    axes[0].legend(loc='upper right',fontsize=8)
    axes[1].plot(trace_t-origin,break_gaps(trace_t,ori),color='#265fa3',linewidth=1.)
    axes[1].set(title='Consumed reference vs waist anchor (policy yaw aligned)',ylabel='SO(3) angle (degrees)',xlabel='Recording time (s)')
    for ax in axes:ax.grid(alpha=.25)
    if report['synthetic']:fig.suptitle('SYNTHETIC TEST DATA - NOT ROBOT PERFORMANCE',color='#a54800')
    fig.savefig(out/'eval_tracking.png',dpi=150);plt.close(fig)
    def fmt(x):return '不可计算' if x is None else f'{x:.3f}'
    table=''.join('<tr><td>'+html.escape(row['comparison'])+'</td><td>'+str(row['frames'])+'</td><td>'+fmt(row['root_fixed_body_error_mm']['mean'])+'</td><td>'+fmt(row['root_fixed_body_error_mm']['p95'])+'</td></tr>' for row in report['execution_shape']['comparisons'])
    world=report['global_reference_tracking']
    if world['status']=='unavailable':
        global_text='暂不可计算：需要同一机器人骨架的独立参考轨迹，以及同步、对齐坐标系的真机全局位姿。缺失值不会填零。'
    else:
        global_text='世界坐标数据状态：'+html.escape(world['status'])+'；MPJPE_G '+fmt(world['mpjpe_g_mm']['mean'])+' mm；MPJPE_L '+fmt(world['mpjpe_l_mm']['mean'])+' mm；根位置 '+fmt(world['root_position_error_mm']['mean'])+' mm；根朝向 '+fmt(world['root_orientation_error_deg']['mean'])+'°。坐标/时间校准以输入文件声明为准。'
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>SONIC 参考与执行评估</title>
<style>body{max-width:1120px;margin:30px auto;padding:0 20px;font:16px system-ui;line-height:1.7;color:#203247}table{border-collapse:collapse;width:100%}td,th{padding:9px;text-align:left;border-bottom:1px solid #ddd}img{width:100%}.note{background:#fff4df;padding:14px}code{overflow-wrap:anywhere}</style>
<h1>SONIC 参考与执行评估</h1>'''
    if report['synthetic']:page+='<p class="note">合成测试数据，非真机成绩。</p>'
    page+='<p><a href="tracking_reference.html">打开新版：重建参考、控制目标、最终指令、实测角度</a> · <a href="report.html">原执行与人体弯曲角报告</a></p>'
    page+='<p>官方 MPJPE_G 看世界坐标中的关节点位置误差；MPJPE_L 只减去各自根位置，仍然包含朝向误差。默认官方回调没有单独计算机器人根朝向误差。</p>'
    page+='<h2>当前能测：执行骨架与参考朝向</h2><p>下表把 torso 固定在相同坐标系，比较目标与实测关节经过同一 URDF 得到的14个骨架点。它衡量执行偏差，不是人体参考的官方 MPJPE_L。</p><table><tr><th>比较</th><th>有效反馈帧</th><th>平均位置误差 mm</th><th>点误差 P95 mm</th></tr>'+table+'</table>'
    page+='<p>参考—锚点旋转误差：平均 '+fmt(report['encoder_anchor_orientation']['mean'])+'°。它来自同一次推理实际输入，已经过控制器 yaw 对齐，不是外部定位得到的绝对朝向误差。</p><img src="eval_tracking.png" alt="执行骨架误差和参考锚点姿态误差曲线">'
    page+='<p class="note">同目录 JSON/CSV 保存指标与曲线；原始记录保留。随包 URDF 尚未核对机器人当前模型；未定义自由遥操的官方 success_rate，也不套用历史仿真合格线。</p></html>'
    page=page.replace('</html>','<details><summary>可选全局评估（不影响局部诊断）</summary><p>'+global_text+'</p></details></html>')
    (out/'eval_report.html').write_text(page)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('recording',type=Path)
    parser.add_argument('--paired-world',type=Path,help='explicit synchronized same-robot world-position NPZ; see README')
    args=parser.parse_args();evaluate(args.recording,args.paired_world)
