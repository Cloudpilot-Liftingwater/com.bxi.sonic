#!/usr/bin/env python3
"""Read downloaded raw files using the existing collector verifier."""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import os
import shutil
import sys
from zoneinfo import ZoneInfo
from timing import analyze_episode, summary_lines

REPO = Path(__file__).resolve().parents[2]
R7 = REPO / "tools/pico_collection/sonic_pico_collection_20261019_r7"
EXPECTED_RELEASE = "20261021-collection-11-official-buttons"


def manifest(folder):
    value = json.loads((folder / "manifest.json").read_text())
    if value.get("schema") != "elf3-collection-download-v1":
        raise ValueError("Unknown manifest")
    allowed = {f"{kind}/{s['name']}" for s in value["sessions"]
               for kind in ("pico_collection_runs", "pico_collection_data")}
    seen = set()
    for item in value["files"]:
        path = Path(item["path"])
        if (path.is_absolute() or ".." in path.parts or len(path.parts) < 3
                or "/".join(path.parts[:2]) not in allowed or str(path) in seen):
            raise ValueError("Unexpected download path")
        seen.add(str(path))
    return value


def prepare(folder):
    value = manifest(folder)
    root = folder / "robot"
    root.mkdir(exist_ok=True)
    total = sum(f["size"] for f in value["files"])
    present = sum(min((root / f["path"]).stat().st_size, f["size"])
                  for f in value["files"] if (root / f["path"]).is_file())
    needed = total - present + max((f["size"] for f in value["files"]), default=0) + 512 * 1024**2
    free = shutil.disk_usage(folder).free
    print(f"DOWNLOAD_GiB={total / 1024**3:.2f}; FREE_GiB={free / 1024**3:.2f}", flush=True)
    if free < needed:
        raise RuntimeError("本地空间不足；已保留清单，未删除任何文件。请换有空间的下载目录。")
    (folder / "files.list").write_bytes(b"".join(
        f["path"].encode() + b"\0" for f in sorted(value["files"], key=lambda f: f["path"])))


def check_hashes(folder, value):
    root = folder / "robot"
    for index, item in enumerate(value["files"], 1):
        path = root / item["path"]
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("Unexpected local link")
        if path.stat().st_size != item["size"]:
            raise ValueError("Incomplete file: " + str(path))
        h = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                h.update(chunk)
        if h.hexdigest() != item["sha256"]:
            raise ValueError("SHA256 mismatch: " + str(path))
        if index % 1500 == 0:
            print(f"LOCAL_HASHING={index}/{len(value['files'])}", flush=True)


def read_json(path):
    return json.loads(path.read_text()) if path.is_file() else {}


def session_evidence(run, selected):
    log = run / "session.log"
    header = ""
    if log.is_file():
        with log.open(errors="replace") as stream:
            header = stream.read(65536)
    release = re.search(r"^COLLECTOR_RELEASE=(.+)$", header, re.M)
    stamp = re.search(r"\[(\d{10}\.\d+)\]", header)
    timestamp = float(stamp[1]) if stamp else selected.get("session_mtime_ns", 0) / 1e9
    return dict(recorded_at=datetime.fromtimestamp(timestamp, ZoneInfo("Asia/Shanghai")).isoformat(),
                time_source="first_ros_log" if stamp else "session_directory_mtime",
                recorded_release=release[1].strip() if release else None)


def review(folder):
    value = manifest(folder)
    # The downloaded robot status identifies the running collector. A workstation
    # constant must not label every newly deployed release as historical.
    expected_release = value.get("current_status", {}).get("collector_release") or EXPECTED_RELEASE
    check_hashes(folder, value)
    print("DOWNLOAD_SHA256_PASS", flush=True)
    sys.path.insert(0, str(R7))
    from verify_saved import verify
    from session_protocol import capture_segment
    results = []
    for selected in reversed(value["sessions"]):
        name = selected["name"]
        run = folder / "robot/pico_collection_runs" / name
        data = folder / "robot/pico_collection_data" / name
        report = read_json(run / "launcher_status.json")
        intent = read_json(run / "intent.json")
        session_result = read_json(run / "session_result.json")
        if (run / 'capture.json').is_file():
            import subprocess
            interpreter = os.environ.get('SONIC_REVIEW_PYTHON', sys.executable)
            tool = R7.parents[1] / 'gr00t_dataset/verify_online.py'
            episode = data / 'episode_0000'
            checked = subprocess.run([str(interpreter), '-B', str(tool), str(episode), '--session', str(run)],
                                     text=True,capture_output=True)
            header = read_json(run/'capture.json')
            evidence = session_evidence(run,selected)
            evidence['recorded_release']=header['collector_release']
            try:
                if checked.returncode: raise ValueError(checked.stderr[-2000:])
                check=json.loads(checked.stdout)
                category=check['category']
                paused=check['pause_resume_verified']
                error=None
            except Exception as exc:
                check={};category='DATA_CHECK_FAILED';paused=False;error=str(exc)
            if evidence['recorded_release']!=expected_release:category='HISTORICAL_SESSION'
            result=dict(session=name,category=category,pause_resume_verified=paused,
                pause_resume_outcome='成功' if paused else ('没有完整暂停续采' if check.get('events') else '没有暂停记录'),
                failure_reason=error or session_result.get('error'),**evidence,
                operator_outcome=session_result.get('outcome'),
                timing=analyze_episode(episode),
                readback_details=check,
                task_labels=([json.loads(s) for s in (episode/'lerobot/meta/tasks.jsonl').read_text().splitlines()]
                             if (episode/'lerobot/meta/tasks.jsonl').is_file() else []),
                training_status='DISCARDED_EXCLUDED' if session_result.get('outcome')=='discarded'
                    else 'RAW_ONLY_EXPORT_AND_LOADER_CHECK_PENDING',
                episodes=[dict(num_frames=check.get('num_frames',0),pauses=[e for e in check.get('events',[]) if e['event']=='pause'],
                    raw_integrity=check.get('readback','FAIL'),local_episode=str(episode),
                    audit={'median_source_hz':check.get('median_source_hz')})])
            results.append(result)
            print('SESSION_RESULT='+json.dumps(result,ensure_ascii=False),flush=True)
            continue
        metas = sorted(data.rglob("meta.json")) if data.exists() else []
        result = dict(session=name, intent_command=intent.get("command"),
                      runtime_failure=report.get("runtime_failure"),
                      exit_code=report.get("exit_code"), session_result=session_result,
                      final_report_present=selected["final_report_present"], episodes=[])
        result.update(session_evidence(run, selected))
        stage = "三路图像/视频逐帧读回中" if metas else "无原始episode，只检查会话日志"
        print(f"REVIEWING={name}; date={result['recorded_at']}; release={result['recorded_release']}; episodes={len(metas)}; {stage}", flush=True)
        for path in metas:
            meta = read_json(path)
            item = dict(local_episode=str(path.parent), num_frames=meta.get("num_frames"),
                        release=meta.get("metadata", {}).get("collector_release"),
                        capture_profile=meta.get("metadata", {}).get("capture_profile"),
                        task_name=meta.get("task_name"),
                        discarded=meta.get("discarded") is True,
                        complete=meta.get("termination", {}).get("success") is True,
                        termination_reason=meta.get("termination", {}).get("reason"),
                        excluded_from_training=meta.get("excluded_from_training"),
                        pauses=meta.get("synchronization", {}).get("pose_pauses", []))
            try:
                check = verify(path.parent, allow_incomplete=True, expected_ports={"left": 7, "right": 1})
                item["audit"] = check
                segments = Counter()
                with (path.parent / "frames.jsonl").open() as stream:
                    for line in stream:
                        row = json.loads(line)
                        segments[capture_segment(float(row["auxiliary"]["anchor_stamp_unix"]), item["pauses"])] += 1
                item["rows_by_segment"] = dict(segments)
                command_pauses = intent.get("pauses", [])
                item["pause_history_matches_intent"] = (
                    len(command_pauses) == len(item["pauses"]) and
                    all(a[1] is not None and b[1] is not None and
                        abs((a[1]-a[0])-(b[1]-b[0])) < .02
                        for a, b in zip(command_pauses, item["pauses"])))
                item["pause_resume_same_episode"] = (
                    bool(item["pauses"]) and all(end is not None for _, end in item["pauses"])
                    and item["pause_history_matches_intent"]
                    and all(segments[i] > 0 and segments[i+1] > 0 for i in range(len(item["pauses"]))))
                actual = {g["side"]: (g["protocol"], g["bus"], g["motor_id"])
                          for g in check.get("gripper_identities", [])}
                item["expected_capture_config"] = (
                    item["release"] == expected_release
                    and item["capture_profile"] == "sonic50_latest_rgb"
                    and actual == {"left": ("bxi", 5, 1), "right": ("bxi", 6, 1)})
                item["raw_integrity"] = check["readback"]
            except Exception as exc:
                item.update(raw_integrity="FAIL", error=str(exc), pause_resume_same_episode=False)
            result["episodes"].append(item)
        episodes = result["episodes"]
        result["failure_reason"] = ((session_result.get("reason") if session_result.get("outcome") == "interrupted" else None)
            or next((e["termination_reason"] for e in episodes if not e["complete"]
                     and not e["discarded"] and e.get("termination_reason")), None)
            or (report.get("runtime_failure") or {}).get("reason"))
        if result["recorded_release"] and result["recorded_release"] != expected_release:
            result["category"] = "HISTORICAL_SESSION"
        elif ((intent.get("command") == "abort" or session_result.get("outcome") == "interrupted"
               or result["failure_reason"]) and len(episodes) == 1
              and not episodes[0]["discarded"] and not episodes[0]["complete"] and episodes[0]["excluded_from_training"] is True
              and episodes[0]["raw_integrity"] in ("PASS", "METADATA_ONLY")):
            result["category"] = "INTERRUPTED_PRESERVED"
        elif (not selected["final_report_present"] or report.get("runtime_failure")
                or report.get("exit_code") not in (0, 2) or len(episodes) != 1):
            result["category"] = "NEEDS_REVIEW"
        elif episodes[0]["raw_integrity"] not in ("PASS", "METADATA_ONLY"):
            result["category"] = "DATA_CHECK_FAILED"
        elif not episodes[0].get("expected_capture_config"):
            result["category"] = "CAPTURE_VERSION_OR_MAPPING_REVIEW"
        elif (episodes[0]["discarded"] and episodes[0]["excluded_from_training"] is True
              and not episodes[0]["complete"] and intent.get("command") == "discard"):
            result["category"] = "DISCARDED_EXCLUDED"
        elif episodes[0]["complete"] and not episodes[0]["discarded"] and intent.get("command") == "finish":
            result["category"] = ("SAVED_PASS" if episodes[0]["audit"]["timing_review"] == "PASS"
                                  else "SAVED_TIMING_REVIEW")
        else:
            result["category"] = "NEEDS_REVIEW"
        result["pause_resume_verified"] = len(episodes) == 1 and episodes[0]["pause_resume_same_episode"]
        if result["pause_resume_verified"]:
            result["pause_resume_outcome"] = "成功"
        elif result["category"] == "INTERRUPTED_PRESERVED":
            result["pause_resume_outcome"] = "未完成（采集中途故障）"
        elif episodes and not any(e["pauses"] for e in episodes):
            result["pause_resume_outcome"] = "没有暂停记录"
        elif episodes and any(e["pauses"] for e in episodes):
            result["pause_resume_outcome"] = "失败（未形成同条暂停前后数据）"
        else:
            result["pause_resume_outcome"] = "记录缺失，无法判断"
        results.append(result)
        print("SESSION_RESULT=" + json.dumps(
            {k: result[k] for k in ("session", "category", "pause_resume_verified", "pause_resume_outcome", "failure_reason")}, ensure_ascii=False), flush=True)
    counts = Counter(r["category"] for r in results)
    # Report operation coverage separately from integrity of the selected data.
    paused_saved = sum(r["pause_resume_verified"] and r["category"].startswith("SAVED_") for r in results)
    paused_discarded = sum(r["pause_resume_verified"] and r["category"] == "DISCARDED_EXCLUDED" for r in results)
    four_case_covered = (len(results) == 4 and counts["SAVED_PASS"] == 2
                        and counts["DISCARDED_EXCLUDED"] == 2
                        and paused_saved >= 1 and paused_discarded >= 1)
    passed = bool(results) and all(
        r["category"] in ("SAVED_PASS", "DISCARDED_EXCLUDED")
        and not r["pause_resume_outcome"].startswith("失败") for r in results)
    summary = dict(status="CURRENT_RELEASE_RAW_CHECK_PASS" if passed else "NEEDS_REVIEW",
                   categories=dict(counts), sessions=results, download_sha256="PASS",
                   expected_release=expected_release,
                   current_release_sessions=sum(r["recorded_release"] == expected_release for r in results),
                   paused_saved=paused_saved, paused_discarded=paused_discarded,
                   two_save_two_discard_pause_coverage=four_case_covered,
                   training_export_tested=False, hardware_commands_sent=0)
    if value.get('selection') == 'current-release':
        summary['selection'] = 'all_current_release_sessions'
    if counts["INTERRUPTED_PRESERVED"]:
        summary["status"] = "FAILED_RECORDING_INTERRUPTED"
    if results and all(r["category"] == "HISTORICAL_SESSION" for r in results):
        summary["status"] = "NO_SESSIONS_FROM_CURRENT_RELEASE"
    (folder / 'TIMING_SUMMARY.json').write_text(json.dumps({r['session']:r['timing']
        for r in results if 'timing' in r},ensure_ascii=False,indent=2)+'\n')
    (folder / "review.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    episode_count = sum(len(r["episodes"]) for r in results)
    lines = [f"采集下载及本地检查：找到{len(results)}次会话、{episode_count}条原始episode", "结果：" + summary["status"],
             f"当前版本：{expected_release}；所选同版本会话数：{summary['current_release_sessions']}。同版本记录也可能跨日期，请按记录时间区分本轮测试。",
             f"保存={counts['SAVED_PASS'] + counts['SAVED_TIMING_REVIEW']}，弃用={counts['DISCARDED_EXCLUDED']}，中断保留={counts['INTERRUPTED_PRESERVED']}；暂停续采后保存={paused_saved}，暂停续采后弃用={paused_discarded}。",
             "文件SHA256：PASS。" + ("已读回原始数值和图像/视频，具体结果见下。" if episode_count else "没有原始数据，未执行图像或数值读回。"),
             "训练状态：本脚本只核验原始数据；仍需核实任务文字、官方清理导出及GR00T加载。",
             ""]
    for r in results:
        lines.append(f"{r['session']}: {r['category']}；暂停后同条续采={r['pause_resume_outcome']}")
        lines.append(f"  记录时间={r['recorded_at']}；采集版本={r['recorded_release']}")
        if r.get('operator_outcome'):
            lines.append('  记录的操作结果='+r['operator_outcome']+'；任务文字='+json.dumps(r['task_labels'],ensure_ascii=False))
        if r.get('timing'):
            lines.extend(summary_lines(r['timing']))
        if r.get("failure_reason"):
            lines.append("  直接失败原因=" + str(r["failure_reason"]))
        for e in r["episodes"]:
            lines.append(f"  帧数={e['num_frames']}，暂停次数={len(e['pauses'])}，分段帧数={e.get('rows_by_segment')}，读回={e['raw_integrity']}")
            if "audit" in e:
                if 'median_source_hz' in e['audit']:
                    lines.append(f"  源记录相邻间隔中位数折算={e['audit']['median_source_hz']} Hz（不是整段吞吐或模型频率）；目录={e['local_episode']}")
                else:
                    lines.append(f"  有效段频率={e['audit'].get('uninterrupted_row_rate_hz')} Hz；目录={e['local_episode']}")
            if e.get("error"):
                lines.append("  错误：" + e["error"])
    (folder / "SUMMARY.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    print("SUMMARY=" + str(folder / "SUMMARY.txt"), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    try:
        (prepare if args.prepare else review)(args.folder.resolve())
    except Exception as exc:
        raise SystemExit("STOP: " + str(exc))
