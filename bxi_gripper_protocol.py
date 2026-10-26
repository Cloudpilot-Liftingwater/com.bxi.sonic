"""Official BXI gripper physical-unit profile. No device access or ROS imports."""
import json
import math
from pathlib import Path
from types import SimpleNamespace

PROFILE = 'bxi_gripper_mm_v1'
CONTRACT_ID = 'elf3_native66_bxi_opening_mm_v2'


def settings():
    value = json.loads(Path(__file__).with_name('bxi_gripper_mm.json').read_text())
    if (value['profile'] != PROFILE or value['contract_id'] != CONTRACT_ID or
            value['position_unit'] != 'mm' or value['effort_unit'] != 'N' or value['aux_enabled'] is not False):
        raise ValueError('BXI mm firmware with default NTC feedback required')
    if value['open_mm'] != 70 or value['closed_mm'] != 0:
        raise ValueError('this release uses confirmed closed 0 / open 70 mm')
    for key in ('kp','kd','speed_mm_s','acceleration_mm_s2','max_vel_register_rad_s','current_limit_a'):
        if isinstance(value[key],bool) or not math.isfinite(value[key]) or value[key] <= 0:
            raise ValueError('invalid BXI parameter '+key)
    if value['kp'] > 5 or value['kd'] > 1:
        raise ValueError('BXI mm gain range is kp<=5, kd<=1')
    # C round for positive values, not Python round-to-even.
    value['velocity_limit_mm_s'] = math.floor(89.5*value['max_vel_register_rad_s']/(2*math.pi)+.5)
    value['force_limit_n'] = value['current_limit_a']*.07/(2*.007125)
    if value['speed_mm_s'] > value['velocity_limit_mm_s']:
        raise ValueError('trajectory speed exceeds configured wire mapping')
    return value


def profile(bus, value):
    return SimpleNamespace(bus=bus,can_id=1,protocol=SimpleNamespace(value='bxi'),
        position_range=(0.,90.),velocity_range=(-value['velocity_limit_mm_s'],value['velocity_limit_mm_s']),
        torque_range=(-value['force_limit_n'],value['force_limit_n']),kp_range=(0.,5.),kd_range=(0.,1.),
        profile=PROFILE,position_unit='mm',effort_unit='N')
