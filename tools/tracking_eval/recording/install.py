"""Apply only a checked observation hook; never stop/launch any robot process."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parent
BASE = Path('/home/bxi/bxi_vla_ws/bxi_rl_controller_ros2_example')
FOLDERS = ('src/bxi_example_py_elf3/mods/com.bxi.sonic',
           'install/share/bxi_example_py_elf3/mods/com.bxi.sonic')
from tracking_compat import patched

# Observe-only hook files are not loaded by the resident gateway/remote control.
# Block the actual controller, input workers and active recording/test programs.
ACTIVE_PATTERN = (
    '[h]ardware_elf3|[b]xi_example_py_elf3_demo|[b]xi_example_demo[.]py|[m]od_node_runner|'
    '[r]os2 (launch|run) bxi_example_py_elf3|[p]ico_manager_thread_server|[m]anager_launcher[.]py|[p]ose_to_smpl_ref_bridge|'
    '[r]eal_robot_collection|[s]ession_node[.]py|[l]aunch_collection[.]py|'
    '[r]ecord[.]py|[t]est_grippers[.]py|[g]ripper_backend|[g]ripper_demo[.]py'
)


def ancestor_pids():
    """Our SSH/bash/sudo wrappers contain the unexecuted record.py branch."""
    ancestors = set()
    pid = os.getpid()
    try:
        while pid:
            if pid in ancestors or len(ancestors) >= 128:
                raise ValueError('invalid parent chain')
            ancestors.add(pid)
            status = Path(f'/proc/{pid}/status').read_text()
            pid = int(next(line.split()[1] for line in status.splitlines() if line.startswith('PPid:')))
    except (OSError, ValueError, StopIteration) as exc:
        raise SystemExit('STOP: cannot inspect installer parent chain; no writes. '+str(exc))
    return ancestors


def require_inactive():
    ancestors = ancestor_pids()
    active=subprocess.run(['pgrep','-af',ACTIVE_PATTERN],capture_output=True,text=True)
    if active.returncode==0:
        blocked = []
        for line in active.stdout.splitlines():
            try:
                pid = int(line.split(None, 1)[0])
            except (ValueError, IndexError):
                raise SystemExit('STOP: malformed process result; no writes.')
            if pid not in ancestors:
                blocked.append(line)
        if blocked:
            raise SystemExit('STOP: 请正常结束自己的控制/记录流程再安装；不会自动停进程。\n'+'\n'.join(blocked))
    elif active.returncode!=1:
        raise SystemExit('STOP: process check failed; no writes. '+getattr(active,'stderr',''))


def plan(base=BASE, remove=False):
    primary = (ROOT/'baseline_policy.py.txt').read_bytes()
    contract = json.loads((ROOT/'contract.json').read_text())
    if hashlib.sha256(primary).hexdigest()!=contract['baseline_policy_sha256']:
        raise ValueError('baseline checksum mismatch')
    baselines = [primary, (ROOT/'legacy_policy.py.txt').read_bytes()]
    if hashlib.sha256(baselines[1]).hexdigest() != contract['legacy_policy_sha256']:
        raise ValueError('legacy baseline checksum mismatch')
    tap = (ROOT/'tracking_tap.py').read_bytes()
    changes = {}
    for folder in FOLDERS:
        folder = Path(base)/folder
        policy = (folder/'policy.py').resolve(strict=True)
        target = (folder/'tracking_tap.py').resolve()
        if not all(p.is_relative_to(Path(base).resolve()) for p in (policy,target)):
            raise ValueError('target points outside VLA workspace')
        before = policy.read_bytes()
        candidates = [(b, patched(b.decode()).encode()) for b in baselines]
        pair = next((pair for pair in candidates if before in pair), None)
        if pair is None:
            raise ValueError('unknown policy modification; refused: '+str(policy))
        original, modified = pair
        desired = original if remove else modified
        if before!=desired:changes[policy]=(before,desired)
        before = target.read_bytes() if target.exists() else None
        if before is not None and before!=tap:
            raise ValueError('unknown tracking_tap.py; refused: '+str(target))
        desired = None if remove else tap
        if before!=desired:changes[target]=(before,desired)
    return changes


def write(path, data):
    if data is None:path.unlink(missing_ok=True);return
    stat = path.stat() if path.exists() else path.parent.stat()
    fd, temporary = tempfile.mkstemp(prefix='.tracking_',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as f:f.write(data);f.flush();os.fsync(f.fileno())
        os.chmod(temporary,0o644)
        if os.geteuid()==0:os.chown(temporary,stat.st_uid,stat.st_gid)
        os.replace(temporary,path)
    finally:
        if os.path.exists(temporary):os.unlink(temporary)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--apply',action='store_true');p.add_argument('--remove',action='store_true')
    p.add_argument('--require-installed',action='store_true',help='read-only gate before recording')
    args=p.parse_args()
    if args.require_installed and (args.apply or args.remove):
        p.error('--require-installed cannot apply or remove files')
    if socket.gethostname().split('.')[0]!='elf3-test-01':raise SystemExit('wrong robot host')
    print('INSTALLER_RELEASE=20261019-tracking-installer-04')
    changes=plan(remove=args.remove)
    if args.require_installed:
        if changes:
            raise SystemExit('TRACE_HOOK_MISSING: 观察点尚未完整安装，本次不启动记录。先 --install 成功，再手动启动原控制流程。')
        print('TRACKING_HOOK_FILES_PRESENT: 文件完整；运行中的控制器仍需实际输出 trace 才算加载成功。')
        return
    for target in changes:print('PLANNED='+str(target))
    if not args.apply:
        print('CHECK_PASS: observation hook only; no files/services/motors changed.')
        print('HOOK_STATUS='+('INSTALL_REQUIRED' if changes else 'INSTALLED_ON_DISK'))
        return
    if not changes:print('ALREADY_APPLIED');return
    require_inactive()
    backup=Path(tempfile.mkdtemp(prefix='sonic_tracking_backup_',dir=BASE.parent))
    entries=[dict(path=str(path),before=None if old is None else old.hex(),after_sha256=None if new is None else hashlib.sha256(new).hexdigest())
             for path,(old,new) in changes.items()]
    (backup/'transaction.json').write_text(json.dumps(entries,indent=2)+'\n')
    applied=[]
    try:
        for path,(old,new) in changes.items():
            actual=path.read_bytes() if path.exists() else None
            if actual!=old:raise ValueError('file changed after check: '+str(path))
            applied.append(path);write(path,new)
        if plan(remove=args.remove):raise ValueError('post-install check failed')
    except BaseException:
        for path in reversed(applied):write(path,changes[path][0])
        raise
    print('BACKUP='+str(backup))
    print('TRACKING_HOOK_REMOVED' if args.remove else 'TRACKING_HOOK_INSTALLED_NOT_HARDWARE_VALIDATED')
    print('模型/控制计算/夹爪参数未修改。手动启动原流程；记录器运行时才输出观察数据。')


if __name__=='__main__':
    main()
