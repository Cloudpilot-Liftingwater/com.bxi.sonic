"""Replay the actual installer false positive; no robot/network/process signals."""
import re
from pathlib import Path
from types import SimpleNamespace as NS
import sys

import pytest
import test_tracking as base
import install


WRAPPER = ('17981 bash -c set -euo pipefail; if [[ "$2" == --record ]]; then '
           '/usr/bin/python3 -B -u record.py --output "$TRACKING_RUN"; else '
           'sudo /usr/bin/python3 -B install.py --apply; fi '
           'sonic-tracking /home/bxi/sonic_tracking_upload.Wi6YUj --install '
           'sonic_tracking_20261015_v3 sha256 60')


def processes(monkeypatch, lines, ancestors=(17981,)):
    monkeypatch.setattr(install, 'ancestor_pids', lambda: set(ancestors))
    def pgrep(command, **kwargs):
        assert command[:2] == ['pgrep', '-af']
        matches = [line for line in lines if re.search(command[2], line)]
        return NS(returncode=0 if matches else 1, stdout='\n'.join(matches), stderr='')
    monkeypatch.setattr(install.subprocess, 'run', pgrep)


def candidate(tmp_path, monkeypatch):
    base.make_install_tree(tmp_path)
    original = install.plan
    monkeypatch.setattr(install, 'BASE', tmp_path)
    monkeypatch.setattr(install, 'plan', lambda remove=False: original(tmp_path, remove=remove))
    monkeypatch.setattr(install.socket, 'gethostname', lambda: 'elf3-test-01')
    return original


def test_reported_own_shell_is_ignored_and_install_completes(tmp_path, monkeypatch):
    tmp_path = tmp_path/'workspace'  # contain the installer's sibling backup in this test
    original = candidate(tmp_path, monkeypatch)
    assert re.search(install.ACTIVE_PATTERN, WRAPPER)  # reproduce the old false match
    processes(monkeypatch, [WRAPPER])
    monkeypatch.setattr(sys, 'argv', ['install.py', '--apply'])
    install.main()
    assert not original(tmp_path)


@pytest.mark.parametrize('other', [
    '17982 python3 -B -u record.py --output /home/bxi/sonic_tracking_runs/run.test',
    '17983 /opt/bxi/bxi_ros2_pkg/lib/hardware_elf3/hardware_elf3',
    '17984 python3 /home/bxi/bxi_vla_ws/bxi_rl_controller_ros2_example/install/lib/bxi_example_py_elf3/bxi_example_py_elf3_demo',
    '17985 python3 -c import runpy /home/bxi/bxi_vla_ws/bxi_rl_controller_ros2_example/src/bxi_example_py_elf3/mods/com.bxi.sonic/pico/manager_launcher.py --manager',
    '17986 python3 /opt/ros/humble/bin/ros2 run bxi_example_py_elf3 bxi_example_py_elf3_demo',
    WRAPPER.replace('17981', '17987'),  # another shell is NOT automatically exempt
])
def test_only_own_ancestors_exempt_others_still_block_without_writes(tmp_path, monkeypatch, other):
    original = candidate(tmp_path, monkeypatch)
    processes(monkeypatch, [WRAPPER, other])
    monkeypatch.setattr(sys, 'argv', ['install.py', '--apply'])
    with pytest.raises(SystemExit, match='正常结束') as error:
        install.main()
    assert other in str(error.value)
    assert '17981 bash' not in str(error.value)
    assert len(original(tmp_path)) == 4
    assert not list(tmp_path.parent.glob('sonic_tracking_backup_*'))


def test_camera_binaries_not_mistaken_for_controller_package(monkeypatch):
    root='/home/bxi/bxi_vla_ws/bxi_rl_controller_ros2_example/src/bxi_example_py_elf3/mods/com.bxi.sonic/'
    processes(monkeypatch, [WRAPPER, '21487 '+root+'runtime/linux-x86_64/mediamtx '+root+'runtime/mediamtx.yml',
                           '21488 '+root+'bin/linux-x86_64/head_camera_rtsp_node --ros-args'])
    install.require_inactive()


def test_real_parent_chain_contains_self_and_immediate_parent():
    ancestors = install.ancestor_pids()
    assert install.os.getpid() in ancestors
    assert install.os.getppid() in ancestors


def test_parent_chain_read_error_refuses_check(monkeypatch):
    def denied(*args, **kwargs): raise PermissionError('synthetic unreadable proc')
    monkeypatch.setattr(Path, 'read_text', denied)
    with pytest.raises(SystemExit, match='cannot inspect installer parent chain'):
        install.require_inactive()


def test_record_gate_refuses_missing_or_partial_hook_without_writes(tmp_path, monkeypatch):
    original = candidate(tmp_path, monkeypatch)
    monkeypatch.setattr(sys, 'argv', ['install.py', '--require-installed'])
    with pytest.raises(SystemExit, match='TRACE_HOOK_MISSING'):
        install.main()
    assert len(original(tmp_path)) == 4
    target, (_, data) = next(iter(original(tmp_path).items()))
    install.write(target, data)
    with pytest.raises(SystemExit, match='TRACE_HOOK_MISSING'):
        install.main()


def test_record_gate_requires_exact_complete_hook_but_not_inactive_controller(tmp_path, monkeypatch, capsys):
    original = candidate(tmp_path, monkeypatch)
    for target, (_, data) in original(tmp_path).items(): install.write(target, data)
    def forbidden(): raise AssertionError('read-only gate must not require idle robot')
    monkeypatch.setattr(install, 'require_inactive', forbidden)
    monkeypatch.setattr(sys, 'argv', ['install.py', '--require-installed'])
    install.main()
    assert 'TRACKING_HOOK_FILES_PRESENT' in capsys.readouterr().out
    assert not original(tmp_path)
    monkeypatch.setattr(sys, 'argv', ['install.py', '--apply'])
    install.main()
    assert 'ALREADY_APPLIED' in capsys.readouterr().out


def test_policy_tap_record_and_analysis_identical_to_v3():
    previous = base.ROOT.parents[1]/'outputs/sonic_tracking_20261015_v3'
    if not previous.exists(): pytest.skip('development baseline only')
    for name in ('tracking_tap.py','tracking_compat.py','baseline_policy.py.txt','legacy_policy.py.txt',
                 'record.py','analyze.py','analysis_core.py','geometry.py','html_report.py','contract.json'):
        assert (base.ROOT/name).read_bytes() == (previous/name).read_bytes(), name
