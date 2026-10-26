"""Read-only source-clock and writer-backlog statistics. No filtering or resampling."""
import json
import math
from pathlib import Path
import statistics


def distribution(values):
    x=sorted(float(v) for v in values if v is not None and math.isfinite(float(v)))
    if not x:return None
    i=(len(x)-1)*.99;lo=int(i);hi=min(lo+1,len(x)-1)
    return dict(count=len(x),median=statistics.median(x),p99=x[lo]+(x[hi]-x[lo])*(i-lo),min=x[0],max=x[-1])


def span_rate(values):
    if len(values)<2 or any(v is None for v in values):return None
    dt=values[-1]-values[0]
    return (len(values)-1)/dt if dt>0 else None


def analyze_episode(path):
    source=Path(path)/'source.jsonl'
    if not source.is_file():return dict(status='NO_SOURCE_TIMESTAMPS')
    with source.open() as stream:
        rows=[json.loads(line) for line in stream if line.strip()]
    if not rows:return dict(status='EMPTY')
    delta={}
    for key in ['state','command','imu','ego_view','left_wrist','right_wrist']:
        # Both stamps are ROS time. Positive means a sensor newer than the token.
        delta[key]=distribution([(r['source_stamps'][key]-r['inference_stamp_unix'])*1000
            for r in rows if r.get('inference_stamp_unix') is not None and r.get('source_stamps',{}).get(key) is not None])
    queue=distribution([r['writer_queue_age_s']*1000 for r in rows if r.get('writer_queue_age_s') is not None])
    receipt=distribution([r['inference_receipt_delay_s']*1000 for r in rows if r.get('inference_receipt_delay_s') is not None])
    source_stamps=[r.get('inference_stamp_unix') for r in rows]
    received=[r.get('received_monotonic') for r in rows]
    writer_starts=[r['received_monotonic']+r['writer_queue_age_s'] if
        r.get('received_monotonic') is not None and r.get('writer_queue_age_s') is not None else None for r in rows]
    gaps=[];sequence_gaps=resets=0
    for a,b in zip(rows,rows[1:]):
        if a.get('inference_sequence') is None or b.get('inference_sequence') is None:continue
        diff=b['inference_sequence']-a['inference_sequence']
        if diff>1:sequence_gaps+=diff-1
        elif diff<0:resets+=1
        if a.get('inference_stamp_unix') is not None and b.get('inference_stamp_unix') is not None and diff==1:
            dt=b['inference_stamp_unix']-a['inference_stamp_unix']
            if dt>0:gaps.append(dt*1000)
    notices=[]
    if any(v and v['max']>=1000 for v in delta.values()):notices.append('SENSOR_MORE_THAN_1S_NEWER_THAN_TOKEN')
    if any(v and v['min']<=-1000 for v in delta.values()):notices.append('SENSOR_MORE_THAN_1S_OLDER_THAN_TOKEN')
    if queue and queue['max']>=1000:notices.append('WRITER_QUEUE_AGE_OVER_1S')
    if receipt and receipt['max']>=1000:notices.append('TOKEN_RECEIPT_DELAY_OVER_1S')
    return dict(status='SECOND_SCALE_DELAY_OBSERVED' if notices else 'NO_SECOND_SCALE_DELAY_OBSERVED',
        rows=len(rows),sensor_minus_token_ms=delta,writer_queue_age_ms=queue,token_receipt_delay_ms=receipt,
        source_span_hz_including_pauses=span_rate(source_stamps),
        receipt_span_hz_including_pauses=span_rate(received),writer_span_hz_including_pauses=span_rate(writer_starts),
        consecutive_sequence_interval_ms=distribution(gaps),sequence_gaps=sequence_gaps,sequence_resets=resets,
        notices=notices,notes=['1s only labels second-scale delays, not a training acceptance threshold',
        'Span rates include pauses; sequence gaps do not by themselves prove transport loss',
        'Absence of second-scale delay is not proof of full synchronization or training readiness'])


def display(value):
    if not value:return '无记录'
    return f"中位={value['median']:.2f}，P99={value['p99']:.2f}，最小={value['min']:.2f}，最大={value['max']:.2f} ms"


def summary_lines(result):
    if 'sensor_minus_token_ms' not in result:return ['  时序：'+result['status']]
    lines=['  时序：'+result['status'],
        '  token接收延迟：'+display(result['token_receipt_delay_ms']),
        '  写入排队：'+display(result['writer_queue_age_ms']),
        '  状态时间－token时间：'+display(result['sensor_minus_token_ms']['state'])]
    for key in ['ego_view','left_wrist','right_wrist']:
        lines.append('  '+key+'画面时间－token时间：'+display(result['sensor_minus_token_ms'][key]))
    lines.append('  时间差正数=状态/画面晚于token；负数=沿用更早数据。不是训练验收阈值。')
    lines.append('  全条跨度频率（含暂停）='+str({k:round(v,2) if v is not None else None for k,v in
        [('源行',result['source_span_hz_including_pauses']),('接收',result['receipt_span_hz_including_pauses']),('写入',result['writer_span_hz_including_pauses'])]})+' Hz')
    return lines
