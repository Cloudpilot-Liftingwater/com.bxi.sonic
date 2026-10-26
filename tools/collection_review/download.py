#!/usr/bin/env python3
"""Read-only SSH/rsync download using the existing manifest and review logic."""
import argparse
import json
import os
from pathlib import Path
import re
import shlex
import socket
import subprocess
import tempfile

from review import manifest, prepare, check_hashes


def run(args):
    if not re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_.-]*@[A-Za-z0-9][A-Za-z0-9.-]*', args.host):
        raise ValueError('--host must be user@hostname or user@IPv4')
    user, name = args.host.split('@', 1)
    addresses = {x[4][0] for x in socket.getaddrinfo(name, 22, socket.AF_INET, socket.SOCK_STREAM)}
    if len(addresses) != 1:
        raise ValueError('Host has multiple IPv4 addresses; specify its current IP explicitly')
    target = user + '@' + addresses.pop()
    alias = args.host_key_alias or name
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9.-]*', alias):
        raise ValueError('Invalid host key alias')
    report = (args.resume or args.output).expanduser().absolute()
    if args.resume:
        value = manifest(report)
        identity = json.loads((report/'connection.json').read_text()) if (report/'connection.json').is_file() else None
        if identity and identity != dict(user=user, host_key_alias=alias):
            raise ValueError('Resume must use the original SSH user and host-key alias')
    else:
        report.mkdir(parents=True, exist_ok=True)
        if any(report.iterdir()):
            raise ValueError('Use an empty output directory, or --resume')
    print('REPORT_DIR=' + str(report), flush=True)
    print('Read-only download; no service changes, recording, ROS or motor commands.', flush=True)
    with tempfile.TemporaryDirectory(prefix='sonic-download-') as socket_dir:
        control = str(Path(socket_dir)/'ssh')
        options = ['-o', 'StrictHostKeyChecking=yes', '-o', 'HostKeyAlias='+alias,
                   '-o', 'ConnectTimeout=10', '-o', 'ServerAliveInterval=15',
                   '-o', 'ServerAliveCountMax=3', '-o', 'ControlMaster=auto',
                   '-o', 'ControlPersist=120', '-o', 'ControlPath='+control]
        ssh = ['ssh', *options, target]
        try:
            # Keep stdin attached to the terminal for SSH authentication first.
            subprocess.run([*ssh, 'true'], check=True)
            if not args.resume:
                command = ['python3', '-B', '-', '--base', args.remote_base]
                if args.all:
                    command.append('--all')
                with Path(__file__).with_name('plan.py').open('rb') as source, (report/'manifest.json.part').open('wb') as out:
                    subprocess.run([*ssh, shlex.join(command)], stdin=source, stdout=out, check=True)
                (report/'manifest.json.part').replace(report/'manifest.json')
                value = manifest(report)
            (report/'connection.json').write_text(json.dumps(dict(user=user, host_key_alias=alias))+'\n')
            base = value['remote_base']
            if not Path(base).is_absolute() or '\n' in base:
                raise ValueError('Invalid remote base in manifest')
            prepare(report)
            subprocess.run(['rsync', '-rt', '--checksum', '--protect-args', '--partial',
                            '--partial-dir=.rsync-partial', '--from0',
                            '--files-from='+str(report/'files.list'), '--info=progress2',
                            '-e', shlex.join(['ssh', *options]),
                            target+':'+base.rstrip('/')+'/', str(report/'robot')+'/'], check=True)
            check_hashes(report, value)
            print('DOWNLOAD_SHA256_PASS', flush=True)
        finally:
            if Path(control).exists():
                subprocess.run(['ssh', '-S', control, '-O', 'exit', target],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print('Next: review.py, export_all.py and validate_exports.py ' + str(report), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--host', required=True, help='SSH user@host; current IP or resolvable hostname')
    p.add_argument('--host-key-alias', help='Previously trusted SSH hostname when connecting by IP')
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument('--output', type=Path, help='New empty workstation directory')
    group.add_argument('--resume', type=Path, help='Existing directory containing manifest.json')
    p.add_argument('--all', action='store_true', help='All sessions, including discarded/interrupted; otherwise latest four')
    p.add_argument('--remote-base', default='~', help='Robot parent directory containing pico_collection_runs and pico_collection_data')
    run(p.parse_args())
