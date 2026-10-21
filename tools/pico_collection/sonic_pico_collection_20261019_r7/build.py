#!/usr/bin/env python3
"""Immutable same-episode pause/resume collector; controller files untouched."""
import hashlib
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parent
PREVIOUS = ROOT.parent/'sonic_pico_collection_20261014_r6'
TARGET = ROOT.parents[1]/'outputs'/f'{ROOT.name}.tgz'
ALLOWED = {'build.py', 'README.md', 'button_supervisor.py', 'session_protocol.py',
    'session_node.py', 'launch_collection.py', 'elf3_pico_collection.service',
    'install_service.py', 'verify_saved.py', 'collector/real_robot_collection/pico.py',
    'collector/real_robot_collection/node.py', 'collector/real_robot_collection/source_node.py',
    'tests/test_buttons.py', 'tests/test_session.py'}
ADDED = {'tests/test_playout_input.py'}


def files(root):
    return {p.relative_to(root).as_posix():p for p in root.rglob('*')
            if p.is_file() and '__pycache__' not in p.parts
            and p.suffix != '.pyc' and p.name != 'SHA256SUMS'}


def main():
    if TARGET.exists():
        raise RuntimeError('immutable package exists; choose a NEW release')
    old, new = files(PREVIOUS), files(ROOT)
    if set(old)-set(new) or set(new)-set(old)-ADDED:
        raise RuntimeError('unexpected release file additions/deletions')
    for name, path in old.items():
        if name not in ALLOWED and path.read_bytes() != new[name].read_bytes():
            raise RuntimeError('unexpected baseline change: '+name)
    for name, path in new.items():
        if path.is_symlink() or not path.resolve().is_relative_to(ROOT):
            raise RuntimeError('unsafe release path: '+name)
    manifest = ROOT/'SHA256SUMS'
    manifest.write_text(''.join(f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {name}\n'
                               for name,p in sorted(new.items())))
    with tarfile.open(TARGET, 'x:gz') as archive:
        for name,p in sorted(dict(new, SHA256SUMS=manifest).items()):
            archive.add(p, arcname=f'{ROOT.name}/{name}', recursive=False)
    digest = hashlib.sha256(TARGET.read_bytes()).hexdigest()
    TARGET.with_suffix('.tgz.sha256').write_text(f'{digest}  {TARGET.name}\n')
    print(f'PACKAGE={TARGET}\nSHA256={digest}')


if __name__ == '__main__':
    main()
