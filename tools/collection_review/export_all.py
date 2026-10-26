"""Batch the existing official exporter; preserve every source and outcome."""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import html
import json
from pathlib import Path
import shutil
import sys
from urllib.parse import quote
from zoneinfo import ZoneInfo

from review import manifest, check_hashes, read_json


def save(path, value):
    temporary = path.with_name(path.name + '.part')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def outcome_of(info, result, episode_meta):
    # Never silently promote a discard or contradictory evidence to saved data.
    if info.get('discarded_episode_indices') or result.get('outcome') == 'discarded':
        return 'discarded'
    if episode_meta.get('discarded'):
        return 'discarded'
    if result.get('outcome') == 'interrupted' or episode_meta.get('incomplete'):
        return 'interrupted'
    if result.get('outcome') == 'saved':
        return 'saved'
    if episode_meta.get('termination', {}).get('success') and not episode_meta.get('excluded_from_training'):
        return 'saved'
    return 'unknown'


def relative(path, report):
    return str(path.relative_to(report))


def link(path, title):
    return f'<a href="{quote(str(path), safe="/")}">{html.escape(title)}</a>'


def write_reports(report, rows, selection):
    counts = dict(Counter(r['outcome'] for r in rows))
    states = dict(Counter(r['export_status'] for r in rows))
    result = dict(selection=selection, outcomes=counts, exports=states, entries=rows,
                  training_ready=False, hardware_commands_sent=0)
    save(report / 'exports.json', result)
    lines = ['下载原始文件SHA256已核对；原始数据未修改。',
             '导出为LeRobot v2.1：Parquet数值、三路MP4、meta元数据。',
             '当前毫米ABI单独导出；早期弧度/旧JPEG记录保留原件，不能冒充毫米合同。',
             '这是人工检查包；未编造任务标签，弃用/中断不混入保存数据。',
             'GR00T读取结果见 loader_results.json；不存在时表示尚未执行。',
             '保存/弃用等计数=' + json.dumps(counts, ensure_ascii=False),
             '导出结果计数=' + json.dumps(states, ensure_ascii=False), '']
    cards = []
    labels = {'saved':'保存', 'discarded':'弃用', 'interrupted':'中断', 'unknown':'结果不明'}
    for r in rows:
        lines.append(f"{r['recorded_at']} {r['session']} {r['episode']}：{labels[r['outcome']]}；{r['export_status']}；{r.get('reason','')}")
        dataset = report / r['export_path'] if r.get('export_path') else report / r['raw_path']
        meta = dataset / 'meta/info.json'
        videos = []
        if meta.is_file():
            info = read_json(meta)
            for v in sorted(dataset.rglob('*.mp4')):
                title = next((k for k in ('ego_view','left_wrist','right_wrist') if k in str(v)), v.name)
                videos.append(f'<div>{html.escape(title)}<video controls preload="none" src="{quote(relative(v, report), safe="/")}"></video></div>')
            files = [link(relative(meta, report), 'info.json'), link(relative(dataset / 'meta/tasks.jsonl', report), '任务文字')]
            files += [link(relative(p,report), p.name) for p in sorted(dataset.rglob('*.parquet'))]
            if (dataset / 'numeric_preview.txt').is_file():
                files.append(link(relative(dataset / 'numeric_preview.txt', report), '数值预览'))
        else:
            files = [link(r['raw_path'], '原始目录')]
        badges = f"{labels[r['outcome']]} · {r['export_status']} · {r.get('frames',0)}行"
        cards.append('<section><h2>'+html.escape(r['session']+' / '+r['episode'])+'</h2><p>'+html.escape(r['recorded_at']+' · '+badges)+'</p><p>'+html.escape(r.get('reason',''))+'</p><p>'+html.escape('任务：'+'；'.join(r.get('tasks', [])))+'</p><p>'+' · '.join(files)+'</p><div class="views">'+''.join(videos)+'</div><button onclick="playGroup(this)">本条三路一起播放</button><p>'+link(r['raw_path'],'未清理原件（暂停段也保留）')+'</p></section>')
    (report/'EXPORT_SUMMARY.txt').write_text('\n'.join(lines)+'\n')
    page = '''<!doctype html><html lang="zh"><meta charset="utf-8"><title>ELF3采集检查</title>
<style>body{font:16px system-ui;background:#f1f4f8;color:#14202e;max-width:1440px;margin:25px auto;padding:0 18px}section{background:white;margin:18px 0;padding:18px;border-radius:10px}.views{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}video{width:100%;background:#111}a{color:#1769ad}button{margin-top:12px;padding:8px}@media(max-width:700px){.views{grid-template-columns:1fr}}</style>
<h1>ELF3 保存、弃用数据检查</h1><p>三路视频、Parquet数值和任务标签。当前毫米数据复用官方清理后导出；暂停前后仍在同一条episode。原件不变。弃用数据仅供检查，未标注记录待补实际任务文字。</p>
<p><a href="EXPORT_SUMMARY.txt">导出清单</a> · <a href="loader_results.json">GR00T读取检查</a> · <a href="exports.json">逐条路径及结果</a></p>
'''+''.join(cards)+'''<script>function playGroup(b){const vs=b.parentElement.querySelectorAll('video');if(!vs.length)return;const t=vs[0].currentTime;vs.forEach(v=>{v.currentTime=t;v.play().catch(()=>{})})}</script></html>'''
    (report/'index.html').write_text(page)


def run(report, tools):
    sys.path.insert(0, str(tools))
    import export_dataset as exporter
    import pyarrow.parquet as pq
    from contract import CONTRACT_ID
    value = manifest(report)
    check_hashes(report, value)
    print('DOWNLOAD_SHA256_PASS', flush=True)
    rows = []
    for session in sorted(value['sessions'], key=lambda s:s['session_mtime_ns']):
        name = session['name']
        run_dir = report / 'robot/pico_collection_runs' / name
        data_dir = report / 'robot/pico_collection_data' / name
        capture = read_json(run_dir/'capture.json')
        result = read_json(run_dir/'session_result.json')
        stamp = capture.get('created_unix') or session['session_mtime_ns']/1e9
        date = datetime.fromtimestamp(stamp, ZoneInfo('Asia/Shanghai')).isoformat()
        info_paths = sorted(data_dir.rglob('lerobot/meta/info.json'))
        if not info_paths:
            rows.append(dict(session=name, episode='—', outcome=outcome_of({},result,{}), recorded_at=date,
                raw_path=relative(data_dir,report), export_status='ARCHIVED_NO_ONLINE_DATASET',
                reason='旧格式或没有完整episode；原件/会话日志已下载，不伪造训练行。'))
            write_reports(report,rows,value['selection'])
            continue
        for meta in info_paths:
            raw = meta.parent.parent
            info = read_json(meta)
            outcome = outcome_of(info,result,read_json(raw.parent/'meta.json'))
            # The original relative directory is a collision-free episode key.
            key = raw.parent.relative_to(data_dir)
            output = report/'exports'/outcome/name/key
            row = dict(session=name, episode=str(key), outcome=outcome, recorded_at=date,
                raw_path=relative(raw,report), contract_id=info.get('script_config',{}).get('contract_id'),
                frames=info.get('total_frames',0), export_status='PENDING', tasks=[])
            rows.append(row)
            try:
                row['tasks'] = [json.loads(line)['task'] for line in (raw/'meta/tasks.jsonl').read_text().splitlines() if line.strip()]
                if row['contract_id'] != CONTRACT_ID:
                    row.update(export_status='ARCHIVED_LEGACY_ABI',reason='早期非毫米合同：保留原始Parquet/MP4，可查看，不混入当前训练合同。')
                    continue
                files = [f for f in value['files'] if f['path'].startswith(str(raw.parent.relative_to(report/'robot'))+'/')]
                source_id = hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()
                if output.exists():
                    if read_json(output/'BATCH_INPUT.json').get('sha256') != source_id:
                        raise ValueError('已有导出与下载清单不一致；不覆盖，请换新导出目录')
                    exported = read_json(output/'EXPORT_RESULT.json')
                    if not exported:
                        raise ValueError('已有导出不完整；保留现场，不覆盖')
                else:
                    needed = 2 * sum(f['size'] for f in files) + 512*1024**2
                    if shutil.disk_usage(report).free < needed:
                        raise OSError('本地剩余空间不足以保留原始数据及导出视频；原件已下载，未删除')
                    print(f'EXPORTING={name}/{key}; outcome={outcome}',flush=True)
                    exported = exporter.export([raw],output,inspection_only=True,include_discarded=outcome!='saved')
                    save(output/'BATCH_INPUT.json',dict(sha256=source_id,source=str(raw),outcome=outcome))
                row.update(export_status='EXPORTED_FOR_REVIEW', export_path=relative(output,report),
                           frames=exported['frames'], task_annotation_complete=all(exporter.task_valid(t) for t in row['tasks']))
                # Explicitly retain operator outcome even for interrupted/unknown records.
                save(output/'meta/collection_outcome.json',dict(outcome=outcome,source=str(raw),
                    excluded_from_training=outcome!='saved',human_review_pending=True))
                preview=[]
                for p in sorted(output.glob('data/*/*.parquet')):
                    table=pq.read_table(p)
                    preview.append(str(table.schema))
                    preview.append(json.dumps(table.slice(0,3).to_pylist(),ensure_ascii=False,indent=2))
                (output/'numeric_preview.txt').write_text('\n'.join(preview)+'\n')
            except Exception as exc:
                row.update(export_status='EXPORT_NOT_COMPLETED',reason=str(exc))
            finally:
                write_reports(report,rows,value['selection'])
                print(json.dumps(row,ensure_ascii=False),flush=True)
    write_reports(report,rows,value['selection'])
    print('HTML_REPORT='+str(report/'index.html'),flush=True)
    print('EXPORT_SUMMARY='+str(report/'EXPORT_SUMMARY.txt'),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('report',type=Path)
    p.add_argument('--tools',type=Path,default=Path(__file__).resolve().parents[1]/'gr00t_dataset')
    a=p.parse_args()
    run(a.report.resolve(),a.tools.resolve())
