"""Standalone offline report, no CDN/network dependency."""
import json
import numpy as np
from geometry import JOINTS, BENDS


def write_html(path, report, times, angles, valid, human, decoder, policy, command, actual, live):
    from analyze import clean
    stride=max(1,int(np.ceil(len(times)/5000)))
    indices=np.arange(0,len(times),stride)
    t=times-times[0] if len(times) else times
    payload=dict(report=report,names=list(JOINTS),bends=list(BENDS),stride=stride,
        t=t[indices],q=[x[indices] for x in angles],valid=valid[indices],
        shape=[x[indices] for x in (human,decoder,policy,command,actual)],live=live[indices])
    data=json.dumps(clean(payload),ensure_ascii=False,separators=(',',':'),allow_nan=False).replace('<','\\u003c')
    template='''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SONIC 真机跟踪报告</title><style>
body{font:15px system-ui,sans-serif;margin:28px auto;max-width:1280px;padding:0 18px;color:#182b3a;background:#f5f7fa}
h1{font-size:27px}h2{font-size:20px}section{background:white;padding:20px;margin:18px 0;border-radius:10px}
select,input,button{font:inherit;padding:7px;margin:4px}canvas{width:100%;height:350px}table{border-collapse:collapse;width:100%;font-size:13px}
th,td{text-align:right;border-bottom:1px solid #e3e8ed;padding:7px}th:first-child,td:first-child{text-align:left}
.muted{color:#52677b}.warning{color:#8a4d00}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px}#legend span{margin-right:18px}#metric{overflow-wrap:anywhere}
</style><h1>SONIC 真机跟踪报告</h1>
<p id="status"></p><p class="muted">单位：度。误差＝实际−目标。原始数据保留全部帧；评分仅使用有效 live 样本。图中选择的显示区间不改变下面的全段统计。</p>
<section><label>视图 <select id="mode"><option value="robot">机器人关节：decoder → 目标 → 指令 → 实际</option><option value="body">人体与机器人：肘膝几何弯曲角</option></select></label>
<label>关节 <select id="joint"></select></label><br>
<label>开始秒 <input id="from" type="number" step="0.1" value="0" style="width:90px"></label>
<label>结束秒 <input id="to" type="number" step="0.1" style="width:90px"></label><button id="reset">全段</button>
<div id="legend"></div><canvas id="curve"></canvas><h2>执行误差曲线</h2><canvas id="error" style="height:180px"></canvas>
<p id="metric"></p><p class="muted" id="decimation"></p></section>
<section><h2>各关节执行误差（有效 SONIC live 样本）</h2><div style="overflow:auto"><table><thead><tr><th>关节</th><th>有效点数</th><th>MAE°</th><th>RMSE°</th><th>偏差°</th><th>P95绝对误差°</th><th>最大绝对误差°</th><th>估计滞后ms</th><th>补偿后RMSE°</th></tr></thead><tbody id="rows"></tbody></table></div>
<p class="muted">滞后为0–250ms、步长5ms的曲线匹配估计。缺少运动激励不给值；落在搜索边界的结果只提示仍未确定。补偿后误差不能替代原始执行误差。</p></section>
<section><h2>数据质量和观察开销</h2><pre id="quality"></pre></section>
<section><h2>指标含义与限制</h2><ul id="limits"></ul>
<p>同目录输出：report.json、joint_metrics.csv、aligned_joint_angles.csv、body_shape_angles.csv、human_smpl_angles.csv、inference_arrays.npz。全部原始包仍在上级 events.jsonl。</p></section>
<script id="data" type="application/json">__DATA__</script><script>
const D=JSON.parse(document.getElementById('data').textContent),$=id=>document.getElementById(id);
const colors=['#8052ba','#c27b00','#2479bd','#dc3846','#168675'];
const fmt=v=>v===null||v===undefined?'—':Number(v).toFixed(2);
$('status').textContent=(D.report.synthetic?'合成测试数据，非真机成绩。 ':'')+(D.report.status==='OBSERVED_DATA_COMPLETE'?'观察数据检查通过':'数据不完整或无有效评分样本')+'；收到反馈 '+D.report.quality.received_state_rows+' 帧，参与评分 '+D.report.quality.scored_state_rows+' 帧。';
if(D.report.status!=='OBSERVED_DATA_COMPLETE')$('status').className='warning';
$('quality').textContent=JSON.stringify(D.report.quality,null,2);
D.report.limitations.forEach(s=>{let li=document.createElement('li');li.textContent=s;$('limits').append(li)});
D.report.joints.forEach(r=>{let tr=document.createElement('tr');let cells=[r.joint,r.n,fmt(r.mae_deg),fmt(r.rmse_deg),fmt(r.bias_deg),fmt(r.p95_abs_deg),fmt(r.max_abs_deg),r.delay.status==='search_boundary'?'≥'+fmt(r.delay.estimated_lag_ms):fmt(r.delay.estimated_lag_ms),fmt(r.delay.lag_compensated_rmse_deg)];cells.forEach(s=>{let td=document.createElement('td');td.textContent=s;tr.append(td)});$('rows').append(tr)});
$('decimation').textContent='图像每 '+D.stride+' 个反馈点显示一个点；数值统计及CSV使用全部有效点。缺帧会断线，不补零。';
function setJoints(){let names=$('mode').value==='robot'?D.names:D.bends;$('joint').replaceChildren();names.forEach((s,i)=>{let o=new Option(s,i);$('joint').add(o)});draw()}
function plot(id,series,labels){const c=$(id),ctx=c.getContext('2d'),ratio=window.devicePixelRatio||1;let w=c.clientWidth,h=c.clientHeight;c.width=w*ratio;c.height=h*ratio;ctx.scale(ratio,ratio);ctx.clearRect(0,0,w,h);
let lo=Number($('from').value),hi=Number($('to').value);if(!(hi>lo))hi=lo+1;
let values=[];series.forEach(a=>a.forEach((v,i)=>{if(v!==null&&Number.isFinite(v)&&D.t[i]>=lo&&D.t[i]<=hi)values.push(v)}));
let ymin=values.length?Math.min(...values):-1,ymax=values.length?Math.max(...values):1;if(ymax-ymin<1){ymin-=.5;ymax+=.5}let pad=.08*(ymax-ymin);ymin-=pad;ymax+=pad;
let x=t=>58+(w-78)*(t-lo)/(hi-lo),y=v=>h-32-(h-50)*(v-ymin)/(ymax-ymin);
ctx.font='12px system-ui';ctx.lineWidth=1;for(let k=0;k<5;k++){let v=ymin+(ymax-ymin)*k/4;ctx.strokeStyle='#e4e9ef';ctx.beginPath();ctx.moveTo(58,y(v));ctx.lineTo(w-20,y(v));ctx.stroke();ctx.fillStyle='#52677b';ctx.fillText(v.toFixed(1),3,y(v)+4)}
for(let k=0;k<6;k++){let t=lo+(hi-lo)*k/5;ctx.fillText(t.toFixed(1)+'s',x(t)-12,h-8)}
series.forEach((a,s)=>{ctx.strokeStyle=colors[s%colors.length];ctx.lineWidth=1.5;ctx.beginPath();let previous=null;for(let i=0;i<a.length;i++){let v=a[i],t=D.t[i];if(v===null||!Number.isFinite(v)||t<lo||t>hi){previous=null;continue}let gap=previous===null||t-D.t[previous]>Math.max(.06,D.stride*.015);if(gap)ctx.moveTo(x(t),y(v));else ctx.lineTo(x(t),y(v));previous=i}ctx.stroke()});
if(!values.length){ctx.fillStyle='#8a4d00';ctx.fillText('此区间无有效数据',90,70)}}
function draw(){let j=Number($('joint').value||0),robot=$('mode').value==='robot';let arrays=robot?D.q:D.shape;
let series=arrays.map(a=>a.map(row=>row[j]));let labels=robot?['decoder缩放/裁剪后','过渡处理后目标','最终发布指令','实测反馈','encoder参考腕角（仅腕）']:['人体参考弯曲角','decoder机器人弯曲角','过渡后机器人弯曲角','指令机器人弯曲角','实测机器人弯曲角'];
$('legend').replaceChildren();labels.forEach((s,i)=>{let span=document.createElement('span');span.style.color=colors[i];span.textContent='● '+s;$('legend').append(span)});
plot('curve',series,labels);let err=robot?D.q[3].map((r,i)=>D.valid[i][j]?r[j]-D.q[2][i][j]:null):D.shape[4].map((r,i)=>D.live[i]&&r[j]!==null&&D.shape[3][i][j]!==null?r[j]-D.shape[3][i][j]:null);plot('error',[err],['实际−指令']);
let m=robot?D.report.joints[j]:D.report.body_shape[j];$('metric').textContent=robot?'全段有效点 '+m.n+'；平均绝对误差 '+fmt(m.mae_deg)+'°；RMSE '+fmt(m.rmse_deg)+'°；偏差 '+fmt(m.bias_deg)+'°；最大绝对误差 '+fmt(m.max_abs_deg)+'°。':'全段弯曲角 RMSE：人体参考→decoder '+fmt(m.human_to_decoder.rmse_deg)+'°；人体参考→实际 '+fmt(m.human_to_measured.rmse_deg)+'°；最终指令→实际 '+fmt(m.command_to_measured.rmse_deg)+'°。';}
$('to').value=D.t.length?D.t[D.t.length-1].toFixed(3):1;
$('mode').onchange=setJoints;$('joint').onchange=draw;$('from').onchange=draw;$('to').onchange=draw;
$('reset').onclick=()=>{$('from').value=0;$('to').value=D.t.length?D.t[D.t.length-1].toFixed(3):1;draw()};window.onresize=draw;setJoints();
</script></html>'''
    path.write_text(template.replace('__DATA__',data))
