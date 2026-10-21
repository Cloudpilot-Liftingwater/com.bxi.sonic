#!/usr/bin/env python3
"""Start only the two owned wrist cameras and one resident observation process."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import yaml

from configure import configuration, add_gripper_arguments, resolve_devices, camera_command, FIXED_CAMERA_SHA256
from delivery_paths import require_site, runs_root, camera_prefix
from official_runtime import python, check_ros
from launch_collection import output, stop


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    add_gripper_arguments(parser)
    args=parser.parse_args()
    require_site()
    lock=(runs_root()/'listener.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    cfg=configuration(args.left,args.right,args.left_id,args.right_id,left_bus=args.left_bus,right_bus=args.right_bus)
    if cfg['metadata']['capture_profile']!='sonic50_latest_rgb':
        raise RuntimeError('Production native66 uses sonic50_latest_rgb; legacy diagnostics use the historical entry')
    check_ros()
    library=camera_prefix()/'lib/libusb_cam.so'
    if hashlib.sha256(library.read_bytes()).hexdigest()!=FIXED_CAMERA_SHA256:
        raise RuntimeError('Unexpected wrist camera binary; no camera started')
    devices=resolve_devices(cfg)
    for device in devices.values():
        result=output(['fuser',device])
        if result.returncode!=1 or result.stderr.strip():raise RuntimeError('Camera occupied or access unknown: '+device)
    folder=Path(tempfile.mkdtemp(prefix='listener_',dir=runs_root()))
    config=folder/'config.yaml';config.write_text(yaml.safe_dump(cfg,sort_keys=False))
    children=[];logs=[]
    try:
        for side,device in devices.items():
            log=(folder/('camera_'+side+'.log')).open('w');logs.append(log)
            children.append(subprocess.Popen(camera_command(side,device),stdout=log,stderr=subprocess.STDOUT,start_new_session=True))
        env=dict(os.environ,PYTHONNOUSERSITE='1',COLLECTION_LISTENER_PID=str(os.getpid()))
        child=subprocess.Popen([python(),'-B','-u',str(Path(__file__).with_name('official_node.py')),
                               '--config',str(config)],env=env,start_new_session=True)
        children.append(child)
        while all(p.poll() is None for p in children):time.sleep(.2)
        raise RuntimeError('Owned camera/collector exited: '+str([(p.pid,p.poll()) for p in children]))
    except KeyboardInterrupt:
        pass
    finally:
        for p in reversed(children):stop(p)
        for log in logs:log.close()
        lock.close()


if __name__=='__main__':
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(KeyboardInterrupt()))
    signal.signal(signal.SIGHUP,lambda *_:(_ for _ in ()).throw(KeyboardInterrupt()))
    main()
