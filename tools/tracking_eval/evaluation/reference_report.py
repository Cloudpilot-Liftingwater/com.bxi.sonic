"""Standalone four-stage interactive angle report, no browser dependencies."""
import json
import numpy as np
from geometry import JOINTS
from analyze import clean

def write_reference_report(path,report,times,reference,degrees,valid):
    stride=max(1,int(np.ceil(len(times)/5000)))
    indices=np.arange(0,len(times),stride)
    # Break graph segments even if downsampling would skip a brief invalid run.
    display_valid=valid[indices].copy()
    for k,i in enumerate(indices):
        start=indices[k-1] if k else i
        display_valid[k]&=bool(valid[start:i+1].all())
        if k:display_valid[k]&=bool(np.all((np.diff(times[start:i+1])>0)&(np.diff(times[start:i+1])<=.06)))
    payload=dict(report=report,names=JOINTS,stride=stride,
        t=(times-times[0])[indices] if len(times) else [],valid=display_valid,
        q=[x[indices] for x in (reference,degrees[:,0],degrees[:,2],degrees[:,3])],
        policy=degrees[indices,1])
    data=json.dumps(clean(payload),ensure_ascii=False,separators=(',',':'),allow_nan=False).replace('<','\\u003c')
    template='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SONIC 局部跟踪：参考与执行</title><style>
body{max-width:1280px;margin:28px auto;padding:0 18px;font:15px system-ui;line-height:1.65;color:#203247;background:#f4f7fa}
section{background:#fff;border-radius:10px;margin:18px 0;padding:20px}h1{font-size:27px}h2{font-size:19px}select,input,button{font:inherit;padding:7px;margin:4px}canvas{width:100%;height:350px}table{width:100%;border-collapse:collapse;font-size:13px}td,th{text-align:right;padding:8px;border-bottom:1px solid #ddd}td:first-child,th:first-child{text-align:left}.warn{background:#fff2d8;padding:14px}.muted{color:#576a7d}pre{white-space:pre-wrap;overflow-wrap:anywhere}#legend span{margin-right:15px}
</style><h1>SONIC 局部跟踪：参考与执行</h1><p id="status"></p>
<p>人体输入 → Encoder → token；旁路 g1_kin 重建参考，主路 g1_dyn 产生控制目标 → 最终指令 → 实测角度。</p>
<p>参考来自训练模型重建；不是独立动作真值。单位：度，误差＝实际−参考/指令。两边根固定的骨架误差单列，不包含整机转身和位移。</p>
<section><label>关节 <select id="joint"></select></label><label>开始秒 <input id="from" type="number" value="0" step="0.1" style="width:80px"></label><label>结束秒 <input id="to" type="number" step="0.1" style="width:80px"></label><button id="reset">全段</button>
<label><input id="policy" type="checkbox">附加：过渡处理后的目标</label><div id="legend"></div><canvas id="angle"></canvas>
<h2>误差：紫色为实际−参考，蓝色为实际−指令</h2><canvas id="error" style="height:200px"></canvas><p id="selected"></p><p class="muted" id="sampling"></p></section>
<section><h2>全段统计：两类误差并列</h2><p>原始 RMSE 始终保留。滞后在同一有效样本集上搜索 0–250 ms，步长5 ms；补偿后 RMSE 仅辅助判断延迟，不替代原始误差。静止、短片段、断帧和重启不会强行给出滞后。</p>
<div style="overflow:auto"><table><thead><tr><th>关节</th><th>比较</th><th>帧数</th><th>MAE°</th><th>原始 RMSE°</th><th>P95°</th><th>滞后 ms</th><th>同样本未补偿 RMSE°</th><th>补偿后 RMSE°</th></tr></thead><tbody id="metrics"></tbody></table></div></section>
<section><h2>局部骨架与数据来源</h2><p id="shape"></p><p>骨架使用同一 ELF3 URDF 做正向运动学；当前现场模型指纹未核对。主曲线只显示有效 live 配对；CSV 保存全部数据及评分标记。显示区间不改变全段统计。</p><details><summary>时间配对、模型指纹和来源</summary><pre id="quality"></pre></details>
<p><a href="eval_report.html">骨架曲线与朝向诊断</a> · <a href="report.html">原始执行与人体弯曲角报告</a></p><p>可导出文件：reference_joint_angles.csv、reference_joint_metrics.csv、kinematic_reference_arrays.npz、eval_report.json。</p></section>
<script type="application/json" id="data">__DATA__</script><script>
const D=JSON.parse(document.getElementById('data').textContent),R=D.report.kinematic_reference_tracking,$=id=>document.getElementById(id);
const colors=['#8153bd','#c27b00','#2479bd','#dc3846','#168675'],labels=['重建参考 g1_kin','控制目标 g1_dyn','最终下发指令','实际关节角','过渡处理后目标'];
const fmt=x=>x===null||x===undefined?'—':Number(x).toFixed(2);
$('status').textContent=(D.report.synthetic?'合成测试数据，非真机成绩。 ':'')+'参考分析状态：'+R.status+'；有效帧 '+R.frames+'。'+(R.reason||'');if(R.status!=='available'||D.report.synthetic)$('status').className='warn';
$('shape').textContent='参考→实际，固定躯干骨架点平均距离：'+fmt(R.reference_fixed_torso_body_error_mm.mean)+' mm；P95：'+fmt(R.reference_fixed_torso_body_error_mm.p95)+' mm。';
$('quality').textContent=JSON.stringify({timing:R.timing,model:R.model,recording_status:D.report.recording_status,invalid_token_frames:R.invalid_token_frames},null,2);
$('sampling').textContent='图像每 '+D.stride+' 个反馈点显示一个；统计与 CSV 使用全部有效点。未来参考帧不参与当前时刻评分。';
D.names.forEach((s,i)=>$('joint').add(new Option(s,i)));$('joint').value=15;
for(const r of R.joints)for(const [key,label,delay] of [['reference_to_measured','参考→实际','reference_delay'],['command_to_measured','指令→实际','command_delay']]){const m=r[key],d=r[delay],tr=document.createElement('tr');for(const s of [r.joint,label,m.n,fmt(m.mae_deg),fmt(m.rmse_deg),fmt(m.p95_abs_deg),(d.status==='search_boundary'?'≥':'')+fmt(d.estimated_lag_ms),fmt(d.zero_lag_rmse_on_same_samples_deg),fmt(d.lag_compensated_rmse_deg)]){const td=document.createElement('td');td.textContent=s;tr.append(td)}tr.title=d.status;$('metrics').append(tr)}
function plot(id,series,palette){const c=$(id),ctx=c.getContext('2d'),ratio=window.devicePixelRatio||1,w=c.clientWidth,h=c.clientHeight;c.width=w*ratio;c.height=h*ratio;ctx.scale(ratio,ratio);let lo=Number($('from').value),hi=Number($('to').value);if(!(hi>lo))hi=lo+1;
let min=Infinity,max=-Infinity;series.forEach(a=>a.forEach((v,i)=>{if(D.valid[i]&&v!==null&&Number.isFinite(v)&&D.t[i]>=lo&&D.t[i]<=hi){min=Math.min(min,v);max=Math.max(max,v)}}));if(!Number.isFinite(min)){ctx.fillStyle='#8a4d00';ctx.fillText('此区间无有效参考配对；查看状态或原始执行报告。',60,70);return}if(max-min<1){min-=.5;max+=.5}let pad=.08*(max-min);min-=pad;max+=pad;const x=t=>60+(w-80)*(t-lo)/(hi-lo),y=v=>h-30-(h-50)*(v-min)/(max-min);ctx.font='12px system-ui';
for(let k=0;k<5;k++){let v=min+(max-min)*k/4;ctx.strokeStyle='#e3e9ef';ctx.beginPath();ctx.moveTo(60,y(v));ctx.lineTo(w-20,y(v));ctx.stroke();ctx.fillStyle='#52677b';ctx.fillText(v.toFixed(1),4,y(v)+4)}for(let k=0;k<6;k++){let t=lo+(hi-lo)*k/5;ctx.fillText(t.toFixed(1)+'s',x(t)-12,h-6)}
series.forEach((a,s)=>{ctx.beginPath();ctx.strokeStyle=palette[s];ctx.lineWidth=1.6;let prev=null;for(let i=0;i<a.length;i++){let v=a[i],t=D.t[i];if(!D.valid[i]||v===null||!Number.isFinite(v)||t<lo||t>hi){prev=null;continue}if(prev===null||t-D.t[prev]>Math.max(.06,D.stride*.015))ctx.moveTo(x(t),y(v));else ctx.lineTo(x(t),y(v));prev=i}ctx.stroke()})}
function draw(){let j=Number($('joint').value),series=D.q.map(a=>a.map(r=>r[j]));if($('policy').checked)series.push(D.policy.map(r=>r[j]));$('legend').replaceChildren();series.forEach((_,i)=>{let span=document.createElement('span');span.style.color=colors[i];span.textContent='● '+labels[i];$('legend').append(span)});plot('angle',series,colors);let err=[0,2].map(k=>D.q[3].map((r,i)=>r[j]===null||D.q[k][i][j]===null?null:r[j]-D.q[k][i][j]));plot('error',err,[colors[0],colors[2]]);let r=R.joints[j];$('selected').textContent='原始 RMSE：参考→实际 '+fmt(r.reference_to_measured.rmse_deg)+'°；指令→实际 '+fmt(r.command_to_measured.rmse_deg)+'°。'}
function reset(){$('from').value=0;$('to').value=D.t.length?D.t[D.t.length-1].toFixed(3):1;draw()}
for(const id of ['joint','from','to','policy'])$(id).onchange=draw;$('reset').onclick=reset;window.onresize=draw;reset();
</script></html>'''
    path.write_text(template.replace('__DATA__',data))
