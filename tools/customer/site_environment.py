"""Print quoted environment assignments, validating declared external setup files."""
import os
import json
from pathlib import Path
import shlex
import platform

if platform.machine() != 'x86_64':
    raise SystemExit('This delivery camera/BXI binary runtime targets x86_64 only.')
config_path = Path(os.environ['SONIC_SITE_CONFIG']).expanduser().resolve(strict=True)
cfg = json.loads(config_path.read_text())
values = {'SONIC_SITE_CONFIG': str(config_path), 'SONIC_SITE_DOMAIN': str(cfg['ros_domain_id'])}
if type(cfg['ros_domain_id']) is not int or not 0 <= cfg['ros_domain_id'] <= 232:
    raise SystemExit('Invalid ros_domain_id')
for key, env in [('ros_setup', 'SONIC_ROS_SETUP'), ('controller_setup', 'SONIC_CONTROLLER_SETUP')]:
    p = Path(cfg[key]).expanduser()
    if not p.is_absolute() or not p.is_file():
        raise SystemExit(f'Configure the declared external dependency {key}: {p}')
    values[env] = str(p)
for key, env in [('rmw_implementation', 'RMW_IMPLEMENTATION'), ('ros_localhost_only', 'ROS_LOCALHOST_ONLY'), ('cyclonedds_uri', 'CYCLONEDDS_URI')]:
    values[env] = str(cfg[key])
print('\n'.join('export '+k+'='+shlex.quote(v) for k, v in values.items()))
