#!/usr/bin/env python3
"""Install this one listener service; never patch/start controller services."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import pwd
import socket
import subprocess
import tempfile
import time

ROOT = Path(__file__).resolve().parent
NAME = 'elf3_pico_collection.service'
UNIT = Path('/etc/systemd/system')/NAME
KNOWN_UNITS = {
    'b5d4fa6c9594a6a2482d0035d7927c7763752b5f92eddbbcc05a2fac744bf86d',  # known r6 unit
    '31ebf4436009b5139c237704346bf4d1bece28fcbc6752ce6961866bcd192996',
    'd801675099532140db6e123266e2b2b7a2d1dd09c93a451c4e7f48dfcbf3c109',

    '06d7159373e52019c55837b653c90918c84f391a561a552981c23ed57dd01e9c',
    '5fa0148ed03fb67f8871caeda0602afefdcef574bd3b25632e10a3efb84f2cf0',
    'd2fbbec78a9bb01344b12d3ae5fa2a81bd2fbdc433050cf19b2312411f8d9fee',
    'd9320cbbf04eb7d55310b8f637cb05de001edc7b0454190807ceeac621ec9bab',
}


def check_unit(source, target=UNIT, *, replace_known=False):
    if target.is_symlink():
        raise RuntimeError('service path is a symlink/mask; refusing overwrite')
    if not target.exists() or target.read_bytes() == Path(source).read_bytes():
        return
    if not replace_known or hashlib.sha256(target.read_bytes()).hexdigest() not in KNOWN_UNITS:
        raise RuntimeError('unrecognized/different listener service; refusing overwrite')


def verify_release(root=ROOT):
    seen = set()
    for line in (root/'SHA256SUMS').read_text().splitlines():
        digest, name = line.split('  ', 1)
        path = root/name
        if (name in seen or Path(name).is_absolute() or '..' in Path(name).parts
                or path.is_symlink() or not path.resolve().is_relative_to(root.resolve())
                or hashlib.sha256(path.read_bytes()).hexdigest() != digest):
            raise RuntimeError('package checksum/path mismatch: '+name)
        seen.add(name)
    required = {'install_service.py', NAME, 'listen.sh', 'start.sh', 'button_supervisor.py',
                'session_protocol.py', 'session_node.py', 'launch_collection.py'}
    if not required <= seen:
        raise RuntimeError('manifest lacks required listener files')


def install_unit(source, target=UNIT, *, replace_known=False):
    check_unit(source, target, replace_known=replace_known)
    payload = Path(source).read_bytes()
    if target.exists():
        if target.read_bytes() == payload:
            return False
        backup = Path(tempfile.mkdtemp(prefix='elf3_pico_collection_backup_', dir=target.parent))
        old = target.read_bytes()
        with (backup/NAME).open('xb') as handle:
            handle.write(old); handle.flush(); os.fsync(handle.fileno())
        fd, candidate = tempfile.mkstemp(prefix='.'+NAME+'.', dir=target.parent)
        try:
            with os.fdopen(fd, 'wb') as handle:
                handle.write(payload); handle.flush(); os.fsync(handle.fileno())
                os.fchmod(handle.fileno(), 0o644)
            if target.is_symlink() or target.read_bytes() != old:
                raise RuntimeError('unit changed during install; retained backup, no overwrite')
            os.replace(candidate, target)
        finally:
            if os.path.exists(candidate): os.unlink(candidate)
        print('LISTENER_UNIT_BACKUP='+str(backup/NAME))
        return True
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644)
    with os.fdopen(fd, 'wb') as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return True


def require_idle_status(path, now, *, started_after=None):
    value = json.loads(Path(path).read_text())
    if started_after is not None and (value['updated_unix'] < started_after or
                                     value.get('collector_release') != ROOT.name or
                                     value.get('phase') != 'WAITING_FOR_BUTTON'):
        raise RuntimeError('waiting for new listener status, not a stale previous run')
    if (not 0 <= now-value['updated_unix'] <= 5 or
            value.get('phase') not in ('WAITING_FOR_Y_A', 'ERROR_WAITING_FOR_NEW_Y_A',
                                      'WAITING_FOR_BUTTON', 'ERROR_WAITING_FOR_NEW_BUTTON')):
        raise RuntimeError('listener busy or status stale; finish/discard and wait for audit before install')


def locked_file(path, uid, gid):
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    handle = os.fdopen(fd, 'a+')
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        os.fchown(fd, uid, gid)
        return handle
    except Exception:
        handle.close()
        raise RuntimeError('collection/listener lock busy; do not delete lock or force-kill')


def install_and_enable():
    account = pwd.getpwnam('bxi')
    runs = Path('/home/bxi/pico_collection_runs')
    if runs.is_symlink():
        raise RuntimeError('runs path must not be a symlink')
    runs.mkdir(mode=0o700, exist_ok=True)
    if runs.stat().st_uid not in (0, account.pw_uid):
        raise RuntimeError('unexpected runs owner')
    os.chown(runs, account.pw_uid, account.pw_gid)
    # Shared by all released launchers: prevents a new camera/recording session
    # while checking/updating this listener; active listeners are never stopped here.
    with locked_file('/home/bxi/.three_rgb_bxi_trial.lock', account.pw_uid, account.pw_gid):
        state = subprocess.run(['systemctl', 'show', NAME, '--property=ActiveState',
                                '--property=FragmentPath', '--property=DropInPaths', '--property=LoadState'],
                               text=True, capture_output=True, timeout=10)
        props = dict(line.split('=', 1) for line in state.stdout.splitlines() if '=' in line)
        if state.returncode and not (state.returncode == 1 and props.get('LoadState') == 'not-found'):
            raise RuntimeError('cannot inspect listener service: '+state.stderr.strip())
        if props.get('LoadState') not in ('loaded', 'not-found'):
            raise RuntimeError('listener masked or unit invalid; not changed')
        if (props.get('DropInPaths') or
                props.get('FragmentPath') not in ('', str(UNIT))):
            raise RuntimeError('unexpected listener unit origin/drop-ins; not changed')
        if props.get('ActiveState') not in ('inactive', 'failed', 'active'):
            raise RuntimeError('listener is transitioning; wait, no changes made')
        if props.get('ActiveState') == 'active':
            require_idle_status(runs/'status.json', time.time())
            # A running listener must be stopped manually, even if its unit is known.
            check_unit(ROOT/NAME, replace_known=True)
            raise RuntimeError('listener active; no service stopped. Finish saving/audit, then stop ONLY elf3_pico_collection.service manually before apply')
        with locked_file(runs/'listener.lock', account.pw_uid, account.pw_gid):
            install_unit(ROOT/NAME, replace_known=True)
            subprocess.run(['systemctl', 'daemon-reload'], check=True)
        # Release listener lock before starting daemon (no automatic recording).
        started_after = time.time()
        subprocess.run(['systemctl', 'enable', '--now', NAME], check=True)
        subprocess.run(['systemctl', 'is-active', '--quiet', NAME], check=True)
    deadline = time.monotonic()+15
    while time.monotonic() < deadline:
        try:
            require_idle_status(runs/'status.json', time.time(), started_after=started_after)
            subprocess.run(['systemctl', 'is-active', '--quiet', NAME], check=True)
            return
        except (OSError, ValueError, RuntimeError, KeyError):
            time.sleep(.25)
    raise RuntimeError('listener enabled but not ready; inspect journal, do not assume installed successfully')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--check', action='store_true')
    action.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    if socket.gethostname().split('.')[0] != 'elf3-test-01':
        raise RuntimeError('expected robot elf3-test-01')
    if ROOT != Path('/home/bxi/sonic_pico_collection_20261019_r7'):
        raise RuntimeError('unexpected release path')
    pwd.getpwnam('bxi')
    verify_release()
    check_unit(ROOT/NAME, replace_known=True)
    for path in ('/home/bxi/sonic_gripper_fastfollow_20260917/environment.sh',
                 '/home/bxi/bxi_vla_ws/bxi_rl_controller_ros2_example/install/setup.bash',
                 '/home/bxi/bxi_vla_ws/real_robot_collection/.ros_local_root/opt/ros/humble/lib/libusb_cam.so'):
        if not Path(path).is_file():
            raise RuntimeError('required runtime absent: '+path)
    check = 'source "$1/environment.sh" && /usr/bin/python3 -B -c "import rclpy, zmq, cv2, yaml, numpy; from communication.msg import ActuatorStates"'
    subprocess.run(['bash', '-c', check, 'pico-dependency-check', str(ROOT)], check=True, timeout=30)
    subprocess.run(['systemd-analyze', 'verify', str(ROOT/NAME)], check=True, timeout=20)
    if args.check:
        print('CHECK_PASS: package/environment/unit valid. No service or controller changed.')
        return
    if os.geteuid() != 0:
        raise RuntimeError('--apply requires sudo only to install/enable the listener service')
    install_and_enable()
    print('LISTENER_INSTALLED: '+NAME)
    print('READY_STATUS='+Path('/home/bxi/pico_collection_runs/status.json').read_text())
    print('PICO_COLLECTION_READY: '+ROOT.name+'; WAITING_FOR_BUTTON; no automatic recording')
    print('Only collection listener installed/upgraded. No controller/SONIC/gripper source or robot-control service changed.')
    print('Release buttons, enter SONIC live POSE: left grip+A starts/saves, left grip+B discards (retained/excluded).')
    print('A+X leaves POSE: pause; returns to fresh calibrated POSE: resume SAME episode. Explicit save/discard ends it.')
    print('Reboot starts idle listener only. No terminal needed per episode, no automatic recording.')
    print('Logs: journalctl -u '+NAME+' -f')
    print('Disable listener only: sudo systemctl disable --now '+NAME)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        raise SystemExit('STOP: '+str(exc))
