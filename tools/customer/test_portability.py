"""Offline regression of delivery paths; never opens devices or ROS sockets."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import venv

REPO=Path(__file__).resolve().parents[2]
R7=REPO/'tools/pico_collection/sonic_pico_collection_20261019_r7'
sys.path[:0]=[str(R7),str(R7/'collector'),str(REPO/'pico')]
import delivery_paths
import configure
import python_runtime


def test_camera_files_follow_relocated_repo():
    for side in ('left','right'):
        command=configure.camera_command(side,'/dev/video42')
        p=Path(command[command.index('--params-file')+1])
        assert p.is_file() and p.is_relative_to(REPO)
    assert delivery_paths.camera_prefix().is_relative_to(REPO)
    assert (delivery_paths.camera_prefix()/'lib/libusb_cam.so').is_file()


def test_unconfirmed_site_cannot_start(monkeypatch,tmp_path):
    profile=json.loads((REPO/'tools/customer/site.example.json').read_text())
    import socket
    profile['robot_hostname']=socket.gethostname().split('.')[0]
    p=tmp_path/'site.json';p.write_text(json.dumps(profile))
    monkeypatch.setenv('SONIC_SITE_CONFIG',str(p));monkeypatch.setenv('ROS_DOMAIN_ID','31')
    import pytest
    with pytest.raises(RuntimeError,match='confirm left/right'):
        delivery_paths.require_site()


def test_venv_remains_venv(tmp_path):
    dest=tmp_path/'python environment'
    venv.EnvBuilder(with_pip=False).create(dest)
    exe=python_runtime._resolve_python(str(dest/'bin/python'))
    result=subprocess.check_output([str(exe),'-c','import sys;print(sys.prefix)'],text=True).strip()
    assert Path(result)==dest


def test_sourced_environment_handles_spaces_and_nounset(tmp_path):
    fake=tmp_path/'external dependencies';fake.mkdir()
    # Model a ROS setup which requires nounset disabled, without loading ROS.
    setup=fake/'setup.bash';setup.write_text(': "$AMENT_TRACE_SETUP_FILES"\n')
    profile=json.loads((REPO/'tools/customer/site.example.json').read_text())
    profile.update(ros_setup=str(setup),controller_setup=str(setup))
    p=tmp_path/'site config.json';p.write_text(json.dumps(profile))
    env=dict(os.environ,SONIC_SITE_CONFIG=str(p));env.pop('AMENT_TRACE_SETUP_FILES',None)
    result=subprocess.run(['bash','-uc','source "$1"; [[ $- == *u* ]]; test -f "$BXI_LOCAL_ROS_PREFIX/lib/libusb_cam.so"; test -f "$SONIC_XRT_SERVICE_DIR/RoboticsServiceProcess"','bash',str(REPO/'tools/customer/environment.sh')],env=env,capture_output=True,text=True)
    assert result.returncode==0,result.stderr


def test_data_dictionary_lists_actual_row_keys():
    import ast
    source=ast.parse((R7/'collector/real_robot_collection/recorder.py').read_text())
    row=next(n.value for n in ast.walk(source) if isinstance(n,ast.Assign) and isinstance(n.value,ast.Dict) and any(isinstance(t,ast.Name) and t.id=='row' for t in n.targets))
    actual={k.value for k in row.keys}
    documented={x['key'] for x in json.loads((REPO/'docs/COLLECTION_FIELDS.json').read_text())['top_level_fields']}
    assert documented==actual
