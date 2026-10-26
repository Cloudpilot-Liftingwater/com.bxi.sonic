"""Interpreter selection regressions; no ROS/SDK initialization or devices."""
import os
from pathlib import Path
import sys
import venv

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'pico'))
import python_runtime as runtime


def test_system_venv_retains_prefix_and_imports(tmp_path):
    env = tmp_path / 'venv with spaces'
    venv.EnvBuilder(with_pip=False).create(env)
    executable = runtime._resolve_python(str(env / 'bin/python'))
    assert executable == env / 'bin/python'
    assert runtime._conda_prefix(executable) is None
    assert runtime._probe(executable, ('json',)) == (True, '')


def test_conda_python_and_derived_venv_rejected(tmp_path):
    conda = tmp_path / 'shared'
    (conda / 'conda-meta').mkdir(parents=True)
    (conda / 'bin').mkdir()
    # Copied launchers and symlinks must both retain their Conda identity.
    executable = conda / 'bin/python'
    executable.symlink_to(sys.executable)
    assert runtime._conda_prefix(executable) == conda
    derived = tmp_path / 'derived'
    (derived / 'bin').mkdir(parents=True)
    (derived / 'pyvenv.cfg').write_text('home = ' + str(conda / 'bin') + '\n')
    (derived / 'bin/python').symlink_to(sys.executable)
    assert runtime._conda_prefix(derived / 'bin/python') == conda
    assert runtime._probe(derived / 'bin/python', ())[0] is False


def test_explicit_conda_refused_without_executing_it(tmp_path, monkeypatch):
    conda = tmp_path / 'shared'
    (conda / 'conda-meta').mkdir(parents=True)
    (conda / 'bin').mkdir()
    executable = conda / 'bin/python'
    executable.symlink_to(sys.executable)
    monkeypatch.setenv('SONIC_PICO_PYTHON', str(executable))
    def must_not_spawn(*args, **kwargs):
        raise AssertionError('Conda executable must not run')
    monkeypatch.setattr(runtime.subprocess, 'run', must_not_spawn)
    with pytest.raises(RuntimeError, match='Conda-based Python'):
        runtime.select_python('test', ())


def test_activation_cannot_leak_conda_into_subprocess(tmp_path, monkeypatch):
    conda = tmp_path / 'shared'
    (conda / 'conda-meta').mkdir(parents=True)
    monkeypatch.setenv('CONDA_PREFIX', str(conda))
    monkeypatch.setenv('CONDA_EXE', str(conda / 'bin/conda'))
    monkeypatch.setenv('PYTHONPATH', str(conda / 'lib'))
    monkeypatch.setenv('PATH', str(conda / 'bin') + ':/usr/bin:/bin')
    monkeypatch.setenv('LD_LIBRARY_PATH', str(conda / 'lib') + ':/opt/ros/humble/lib')
    cleaned = runtime._clean_python_environment()
    assert 'CONDA_PREFIX' not in cleaned and 'CONDA_EXE' not in cleaned
    assert 'PYTHONPATH' not in cleaned
    assert cleaned['PATH'] == '/usr/bin:/bin'
    assert cleaned['LD_LIBRARY_PATH'] == '/opt/ros/humble/lib'
    assert os.environ['CONDA_PREFIX'] == str(conda)  # Parent shell is untouched.


def test_discovery_uses_project_venv_not_conda(tmp_path, monkeypatch):
    monkeypatch.setenv('BXI_MOD_ROOT', str(tmp_path / 'mod'))
    conda = tmp_path / 'shared'
    (conda / 'conda-meta').mkdir(parents=True)
    (conda / 'bin').mkdir()
    (conda / 'bin/python').symlink_to(sys.executable)
    env = tmp_path / 'mod/.runtime' / runtime.runtime_platform_tag() / 'pico'
    venv.EnvBuilder(with_pip=False).create(env)
    monkeypatch.setattr(runtime.sys, 'executable', str(conda / 'bin/python'))
    monkeypatch.setenv('VIRTUAL_ENV', str(conda))
    monkeypatch.setenv('CONDA_PREFIX', str(conda))
    monkeypatch.setenv('PATH', str(conda / 'bin') + ':/usr/bin')
    candidates = runtime._common_candidates()
    assert candidates[0] == env / 'bin/python'
    assert all(runtime._conda_prefix(p) is None for p in candidates)
    monkeypatch.delenv('SONIC_PICO_PYTHON', raising=False)
    assert runtime.select_python('test', ('json',)).executable == env / 'bin/python'
