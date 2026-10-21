#!/usr/bin/env python3
"""Only own camera/collector processes; never launch controller, CAN or head camera."""
import fcntl
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import yaml

from configure import configuration, resolve_devices, camera_command, FIXED_CAMERA_SHA256, add_gripper_arguments
from delivery_paths import runs_root, data_root, camera_prefix, require_site, site
from check_config import validate_config
from trial_lifecycle import wait_for_trial, audit_after_stop, child_failure


def output(args):
    return subprocess.run(args, text=True, capture_output=True, timeout=10)


def stop(child):
    if child.poll() is not None:
        return
    try:
        os.killpg(child.pid, signal.SIGINT)
    except ProcessLookupError:
        return
    # Never kill a collector that may be flushing saved frames.
    while child.poll() is None:
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            print(f'Waiting for owned child {child.pid} to flush/exit; no forced kill.', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--session-dir', type=Path)
    add_gripper_arguments(parser)
    args = parser.parse_args()
    session = args.session_dir
    if session is not None:
        if (not session.is_dir() or session.is_symlink() or session.resolve() != session
                or session.parent != runs_root()
                or not session.name.startswith('session_')
                or session.stat().st_uid != os.getuid() or session.stat().st_mode & 0o077):
            raise RuntimeError('requires a private owned configured runs/session_* directory')
        from session_protocol import read_intent
        read_intent(session)
    require_site()
    lock = (Path.home()/'.three_rgb_bxi_trial.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    # Keep the historic lock/namespace shared across all releases. Also exclude
    # an idle PICO listener: one button must never launch two independent collectors.
    listener_lock = None
    if session is None:
        runs = runs_root()
        runs.mkdir(parents=True, exist_ok=True)
        listener_lock = (runs/'listener.lock').open('a')
        try:
            fcntl.flock(listener_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('PICO collection listener already active; stop only that listener normally before this trial')
    cfg = configuration(args.left, args.right, args.left_id, args.right_id,
                        left_bus=args.left_bus, right_bus=args.right_bus)
    if session is not None:
        cfg['dataset']['output_root'] = str(data_root())
    validate_config(cfg, formal=False, allow_placeholders=False)
    export_enabled = site().get('training_export', False)
    if export_enabled:
        raise ValueError(
            'Automatic training export is not enabled here. Set training_export=false; '
            'after saving, run tools/gr00t_dataset/export_dataset.py on the verified episode. '
            'The sonic50_latest_rgb profile captures real inference rows with cached 30Hz images.'
        )
    if cfg['metadata'].get('storage_backend') == 'nvidia_sonic_lerobot_v1':
        from official_runtime import python as writer_python
        collector_python = writer_python()
    else:
        collector_python = sys.executable
    library = camera_prefix()/'lib/libusb_cam.so'
    if hashlib.sha256(library.read_bytes()).hexdigest() != FIXED_CAMERA_SHA256:
        raise RuntimeError('fixed usb_cam clock library is not installed')
    if output(['pgrep','-af','[r]eal_robot_collection[.](node|source_node)']).returncode == 0:
        raise RuntimeError('another collector is running; no process stopped')
    topics = output(['ros2','topic','list','--no-daemon'])
    if topics.returncode or '/elf3/recording_status' in topics.stdout or '/bxi_trial/' in topics.stdout:
        raise RuntimeError('collector namespace busy or ROS inspection failed')
    devices = resolve_devices(cfg)
    if not shutil.which('fuser'):
        raise RuntimeError('fuser is required to avoid duplicate camera owners')
    for side, device in devices.items():
        busy = output(['fuser',device])
        if busy.returncode != 1 or busy.stderr.strip():
            raise RuntimeError(f'{device} busy or ownership check failed; no process stopped')
        cfg['cameras'][side+'_wrist_rgb']['resolved_video_device'] = device
    if shutil.disk_usage(data_root()).free < 3*1024**3:
        raise RuntimeError('need 3 GiB free for this trial')
    root = session or Path(tempfile.mkdtemp(prefix='three_rgb_gripper_run_',dir=runs_root()))
    config = root/'trial.yaml'
    with config.open('x') as handle:
        handle.write(yaml.safe_dump(cfg,allow_unicode=True,sort_keys=False))
    data = Path(cfg['dataset']['output_root'])/root.name/'three_rgb_gripper_source'
    episode = data/'episode_0000'
    print(f'RUN_DIR={root}\nDATA_ROOT={data}',flush=True)
    print('COLLECTOR_RELEASE='+cfg['metadata']['collector_release'],flush=True)
    print('CAPTURE_PROFILE='+cfg['metadata']['capture_profile']+'; numerical_fps='+str(cfg['dataset']['fps'])+'; camera_fps=30',flush=True)
    print('TASK_INSTRUCTION='+cfg['dataset']['tasks'][0]['instruction'],flush=True)
    print('EXPECTED_GRIPPERS='+json.dumps(cfg['gripper']['expected_identities']),flush=True)
    print('Gripper source-history v2: preserve valid history; audit invalid intervals; separate transport latency.',flush=True)
    print('WRIST_PORTS='+json.dumps(cfg['metadata']['physical_ports'])+'; existing head RGB. Official SONIC online Parquet/MP4; task labels and training statistics finalized on workstation.',flush=True)
    print('PICO session requested: left grip+A saves, left grip+B discards; A+X pauses/resumes the same episode; max 600 active seconds; no fault retry.' if session else
          'Keep existing SONIC/PICO live. Standalone diagnostic: use the ROS collection control entry; production uses listen.sh and left grip+A/B.',flush=True)
    print('No motion commands, head driver launch, services or deployed config changes.',flush=True)
    children, logs, failure = [], [], None
    try:
        if resolve_devices(cfg) != devices:
            raise RuntimeError('USB topology changed; rerun')
        for side, device in devices.items():
            log = (root/f'camera_{side}.log').open('w'); logs.append(log)
            process = subprocess.Popen(camera_command(side,device),stdout=log,
                stderr=subprocess.STDOUT,start_new_session=True)
            children.append(dict(name=side+'_wrist_camera', process=process, log=str(root/f'camera_{side}.log')))
        time.sleep(2)
        failure = child_failure(children)
        if failure is None:
            command = ([collector_python,'-B','-u',str(Path(__file__).parent/'session_node.py'),
                        '--session-dir',str(session)] if session else
                       [collector_python,'-B','-u','-m','real_robot_collection.source_node'])
            process = subprocess.Popen(command+['--config',str(config),'--data-id',root.name],start_new_session=True,
                env=dict(os.environ, PYTHONNOUSERSITE='1'))
            children.append(dict(name='collector',process=process,log='launcher terminal/local log'))
            if session:
                from session_protocol import wait_for_session
                failure = wait_for_session(episode,data,children,session)
            else:
                failure = wait_for_trial(episode,data,children)
    except KeyboardInterrupt:
        failure = dict(reason='operator_or_session_interrupt')
    except Exception as exc:
        failure = dict(reason='launcher_exception', error=str(exc))
    finally:
        if failure:
            print('CAPTURE_STOP='+json.dumps(failure,ensure_ascii=False),flush=True)
        # Signal only owned children. Collector flushes before camera cleanup.
        for child in reversed(children):
            stop(child['process'])
        for log in logs:
            log.close()
    # Recheck final files AFTER shutdown: a camera can die while the collector
    # is already finalizing. Still report writer outcome, with readback deferred.
    result_file = session/'session_result.json' if session else None
    session_result = json.loads(result_file.read_text()) if result_file and result_file.is_file() else None
    report = audit_after_stop(episode,data,failure,session_result,full_readback=False)
    (root/'launcher_status.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    return report['exit_code']


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    signal.signal(signal.SIGHUP, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        print('STOP: '+str(exc),flush=True)
        raise SystemExit(1)
