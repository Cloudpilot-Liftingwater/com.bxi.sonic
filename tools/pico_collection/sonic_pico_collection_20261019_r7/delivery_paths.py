"""Deployment paths only. No ROS, camera or robot commands at import time."""
import json
import os
from pathlib import Path
import socket

REPO = Path(__file__).resolve().parents[3]

def site():
    filename = os.environ.get('SONIC_SITE_CONFIG')
    path = Path(filename).expanduser() if filename else REPO/'tools/customer/site.example.json'
    cfg = json.loads(path.read_text())
    ports = cfg.get('wrist_usb_ports', {})
    if set(ports) != {'left', 'right'} or any(type(v) is not int or v < 1 for v in ports.values()) or len(set(ports.values())) != 2:
        raise ValueError('wrist_usb_ports requires two distinct positive USB port numbers')
    if type(cfg.get('ros_domain_id')) is not int or not 0 <= cfg['ros_domain_id'] <= 232:
        raise ValueError('invalid ros_domain_id')
    return cfg

def runs_root():
    return Path(site().get('runs_root') or Path.home()/'pico_collection_runs').expanduser().resolve()

def data_root():
    return Path(site().get('data_root') or Path.home()/'pico_collection_data').expanduser().resolve()

def camera_prefix():
    return REPO/'tools/customer/runtime/ros_humble_x86_64'

def require_site():
    if not os.environ.get('SONIC_SITE_CONFIG'):
        raise RuntimeError('set SONIC_SITE_CONFIG to the reviewed site JSON; see tools/customer/README.md')
    cfg = site()
    if socket.gethostname().split('.')[0] != cfg['robot_hostname'] or os.environ.get('ROS_DOMAIN_ID') != str(cfg['ros_domain_id']):
        raise RuntimeError('hostname/ROS_DOMAIN_ID differs from site profile')
    if cfg.get('camera_roles_confirmed') is not True or not cfg.get('camera_role_evidence'):
        raise RuntimeError('confirm left/right wrist camera roles in site profile before recording')
    for path in (runs_root(), data_root()):
        path.mkdir(parents=True, exist_ok=True)
