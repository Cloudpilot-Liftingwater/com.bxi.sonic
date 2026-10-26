#!/usr/bin/env python3
"""Read selected robot collection sessions. stdout is a JSON manifest."""
import contextlib
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import stat
import subprocess
import sys
import time


def read_json(path):
    return json.loads(path.read_text()) if path.is_file() else None


def fingerprint(path):
    before = path.stat()
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError("文件还在写入，请等保存/审计完成：" + str(path))
    return dict(size=after.st_size, sha256=h.hexdigest())


def inventory(root):
    paths = []
    if not root.exists():
        return paths
    if root.is_symlink():
        raise ValueError("Unexpected symlink: " + str(root))
    for directory, dirs, files in os.walk(root, followlinks=False):
        for name in dirs:
            if (Path(directory) / name).is_symlink():
                raise ValueError("Unexpected directory symlink")
        for name in files:
            p = Path(directory) / name
            if not stat.S_ISREG(p.lstat().st_mode):
                raise ValueError("Unexpected non-regular file: " + str(p))
            paths.append(p)
    return sorted(paths)


def build(base, selection="latest-four"):
    runs, data = base / "pico_collection_runs", base / "pico_collection_data"
    state = read_json(runs / "status.json") or {}
    folders = sorted(
        (p for p in runs.glob("session_*") if p.is_dir() and not p.is_symlink()),
        key=lambda p: (p.stat().st_mtime_ns, p.name), reverse=True)
    if selection == "current-release":
        release = state.get("collector_release")
        if not release:
            raise RuntimeError("当前状态没有版本号，无法选择本版记录。")
        folders = [p for p in folders
                   if (read_json(p / "capture.json") or {}).get("collector_release") == release]
    elif selection == "latest-four":
        folders = folders[:4]
    elif selection == "all":
        # Also preserve data whose launcher report was lost or never completed.
        names = {p.name for root in (runs, data) for p in root.glob("session_*")
                 if p.is_dir() or p.is_symlink()}
        folders = [runs / name if (runs / name).exists() else data / name for name in names]
        folders.sort(key=lambda p: (p.stat().st_mtime_ns, p.name), reverse=True)
    else:
        raise ValueError("Unknown session selection")
    if not folders:
        raise RuntimeError("没有找到采集会话目录。")
    sessions, paths = [], []
    for folder in folders:
        if (state.get("session") == str(folder) and
                state.get("phase") not in ("WAITING_FOR_BUTTON", "ERROR_WAITING_FOR_NEW_BUTTON",
                                           "LISTENER_STOPPED")):
            raise RuntimeError("最近一条仍在采集/保存/审计，等COLLECTION_FINISHED再下载。")
        session_files = inventory(runs / folder.name)
        data_files = inventory(data / folder.name)
        paths.extend(session_files + data_files)
        sessions.append(dict(name=folder.name, remote_session=str(runs / folder.name),
                             session_mtime_ns=folder.stat().st_mtime_ns,
                             final_report_present=(runs / folder.name / "launcher_status.json").is_file(),
                             run_directory_present=(runs / folder.name).is_dir(),
                             raw_files=len(data_files)))
    result = dict(schema="elf3-collection-download-v1", observed_unix=time.time(),
                  remote_base=str(base), current_status=state, sessions=sessions, selection=selection,
                  remote_hostname=socket.gethostname(),
                  files=[], hardware_commands_sent=0, source_mutations=0)
    print(f"SELECTED_SESSIONS={len(sessions)} FILES={len(paths)}", file=sys.stderr, flush=True)
    for index, path in enumerate(paths, 1):
        result["files"].append(dict(path=str(path.relative_to(base)), **fingerprint(path)))
        if index % 1500 == 0:
            print(f"READ_ONLY_HASHING={index}/{len(paths)}", file=sys.stderr, flush=True)
    result["total_bytes"] = sum(f["size"] for f in result["files"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="~")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--current-release", action="store_true")
    args = parser.parse_args()
    base = Path(args.base).expanduser().resolve(strict=True)
    # Existing lock only: neither create/delete a lock nor stop a running recorder.
    with contextlib.ExitStack() as stack:
        path = base / ".three_rgb_bxi_trial.lock"
        if path.exists():
            handle = stack.enter_context(path.open("rb"))
            try:
                fcntl.flock(handle, fcntl.LOCK_SH | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError("采集/保存/审计仍在运行，等COLLECTION_FINISHED再下载。")
        selection = "all" if "--all" in sys.argv[1:] else "current-release" if "--current-release" in sys.argv[1:] else "latest-four"
        result = build(base, selection)
        status = subprocess.run(
            ["systemctl", "show", "elf3_pico_collection.service", "-p", "ActiveState",
             "-p", "UnitFileState", "-p", "MainPID", "-p", "ExecStart"],
            capture_output=True, text=True, timeout=10)
        result["service"] = status.stdout
        result["service_read_error"] = status.stderr if status.returncode else None
        json.dump(result, sys.stdout, ensure_ascii=False)
        sys.stdout.write("\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("STOP: " + str(exc), file=sys.stderr)
        raise SystemExit(1)
