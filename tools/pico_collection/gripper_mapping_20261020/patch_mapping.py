#!/usr/bin/env python3
"""Swap only the confirmed test-01 trigger/collector bus mapping. No motion."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import tempfile

import yaml

BASE = Path('/home/bxi/bxi_vla_ws/bxi_rl_controller_ros2_example')
MODS = ('src/bxi_example_py_elf3/mods/com.bxi.sonic',
        'install/share/bxi_example_py_elf3/mods/com.bxi.sonic')
UNIT = Path('/etc/systemd/system/elf3_pico_collection.service')
DEMO_SHA = '37d5432f617955ccc42545708ff2634844492e857e90ca70f4d6eeec630b19a8'
UNIT_SHA = '1616ea6beda63f4f44a8428da981af84a81efcab5df1603e434b71d62f3393e7'
OLD_ARGS = '--left-bus 6 --right-bus 5'
NEW_ARGS = '--left-bus 5 --right-bus 6'
ACTIVE = (
    '[h]ardware_elf3|[b]xi_example_py_elf3_demo|[b]xi_example_demo[.]py|'
    '[m]od_node_runner|[p]ico_manager_thread_server|[p]ose_to_smpl_ref_bridge|'
    '[b]utton_supervisor[.]py|[s]ession_node[.]py|[l]aunch_collection[.]py|'
    '[r]ecord[.]py|[t]est_grippers[.]py|[g]ripper_backend|[g]ripper_demo[.]py'
)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def patch_mod(raw):
    before = yaml.safe_load(raw)
    expected = copy.deepcopy(before)
    params = expected['states']['sonic_teleop']['params']
    if params.get('gripper_backend') != 'bxi_demo' or params.get('hardware_gripper') is not True:
        raise RuntimeError('not the audited, enabled BXI demo configuration')
    if {params['gripper_left_bus'], params['gripper_right_bus']} != {5, 6}:
        raise RuntimeError('unexpected bus pair')
    text = raw.decode()
    for side, bus in (('left', 5), ('right', 6)):
        key = 'gripper_' + side + '_bus'
        # Preserve all unrelated bytes, including comments and formatting.
        text, count = re.subn(r'(?m)^([ \t]*' + key + r':[ \t]*)[56]([ \t]*(?:#[^\n]*)?)$',
                             lambda m: m[1] + str(bus) + m[2], text)
        if count != 1:
            raise RuntimeError('missing/duplicate/nonstandard mapping: ' + key)
        params[key] = bus
    result = text.encode()
    if yaml.safe_load(result) != expected:
        raise RuntimeError('configuration changed outside the two mapping values')
    return result


def patch_unit(raw):
    # Accept the audited r7 unit, or exactly that unit with this patch applied.
    old = raw.replace(NEW_ARGS.encode(), OLD_ARGS.encode())
    if sha(old) != UNIT_SHA or old.count(OLD_ARGS.encode()) != 1:
        raise RuntimeError('collector unit differs from audited r7; refusing overwrite')
    return old.replace(OLD_ARGS.encode(), NEW_ARGS.encode())


def make_plan(base=BASE, unit=UNIT):
    base = base.resolve(strict=True)
    plan = {}
    for relative in MODS:
        folder = base / relative
        if sha((folder / 'bxi_demo.py').read_bytes()) != DEMO_SHA:
            raise RuntimeError('BXI demo changed since audit: ' + str(folder))
        path = (folder / 'mod.yaml').resolve(strict=True)
        if not path.is_relative_to(base):
            raise RuntimeError('SONIC config resolves outside workspace')
        raw = path.read_bytes()
        plan[path] = (raw, patch_mod(raw))
    if unit.is_symlink():
        raise RuntimeError('collector service is a link/mask')
    raw = unit.read_bytes()
    plan[unit] = (raw, patch_unit(raw))
    return plan


def check_runtime(*, idle):
    state = subprocess.run(
        ['systemctl', 'show', UNIT.name, '--no-pager',
         '--property=ActiveState,LoadState,FragmentPath,DropInPaths'],
        capture_output=True, text=True, check=True, timeout=10)
    props = dict(line.split('=', 1) for line in state.stdout.splitlines() if '=' in line)
    if (props.get('LoadState') != 'loaded' or props.get('FragmentPath') != str(UNIT)
            or props.get('DropInPaths')):
        raise RuntimeError('collector has unexpected unit origin/drop-ins')
    print('COLLECTOR_STATE=' + props.get('ActiveState', 'unknown'))
    if not idle:
        return
    if props.get('ActiveState') not in ('inactive', 'failed'):
        raise RuntimeError('先保存当前采集，再手动停止采集监听器；脚本不会停服务。')
    processes = subprocess.run(['pgrep', '-af', ACTIVE], capture_output=True, text=True)
    if processes.returncode != 1:
        raise RuntimeError('控制/输入/记录/测试仍在运行或检查失败，请正常结束自己的流程：\n'
                           + processes.stdout + processes.stderr)


def atomic_write(path, raw):
    stat = path.stat()
    fd, candidate = tempfile.mkstemp(prefix='.gripper_mapping_', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
            if os.geteuid() == 0:
                os.fchown(handle.fileno(), stat.st_uid, stat.st_gid)
            os.fchmod(handle.fileno(), stat.st_mode & 0o777)
        os.replace(candidate, path)
    finally:
        Path(candidate).unlink(missing_ok=True)


def apply_plan(plan, backup_parent, reload_unit):
    changed = {p: pair for p, pair in plan.items() if pair[0] != pair[1]}
    if not changed:
        reload_unit()
        return None
    backup = Path(tempfile.mkdtemp(prefix='gripper_mapping_backup_', dir=backup_parent))
    rows = []
    for i, (path, (before, after)) in enumerate(changed.items()):
        (backup / str(i)).write_bytes(before)
        rows.append(dict(path=str(path), backup=str(i), before=sha(before), after=sha(after)))
    (backup / 'manifest.json').write_text(json.dumps(rows, indent=2) + '\n')
    print('BACKUP=' + str(backup), flush=True)
    done = []
    try:
        for path, (before, _) in plan.items():
            if path.is_symlink() or path.read_bytes() != before:
                raise RuntimeError('file changed after inspection: ' + str(path))
        for path, (before, after) in changed.items():
            atomic_write(path, after)
            done.append((path, before))
        reload_unit()
        for path, (_, after) in plan.items():
            if path.read_bytes() != after:
                raise RuntimeError('post-write verification failed: ' + str(path))
    except Exception:
        for path, before in reversed(done):
            atomic_write(path, before)
        reload_unit()
        raise
    return backup


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--check', action='store_true')
    action.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    if socket.gethostname().split('.')[0] != 'elf3-test-01':
        raise RuntimeError('expected robot elf3-test-01')
    plan = make_plan()
    for path, (before, after) in plan.items():
        print(('WRITE=' if before != after else 'ALREADY_CORRECT=') + str(path))
    check_runtime(idle=args.apply)
    if args.check:
        print('CHECK_PASS: left trigger CAN5, right trigger CAN6; no target files changed.')
        return
    if os.geteuid() != 0:
        raise RuntimeError('--apply requires sudo for the collector unit')
    check_runtime(idle=True)
    apply_plan(plan, BASE.parent, lambda: subprocess.run(
        ['systemctl', 'daemon-reload'], check=True, timeout=30))
    print('MAPPING_INSTALLED: left=CAN5 right=CAN6; controller + collector agree.')
    print('No service started/stopped; no motor commands. Restart your original workflow manually.')


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        raise SystemExit('STOP: ' + str(error))
