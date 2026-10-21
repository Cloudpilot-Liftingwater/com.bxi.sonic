"""Explicit non-training mode. Does not synthesize cameras or training coordinates."""
import math
import numpy as np
from .gripper_identity import identities, validate_identities, check_sample_identity

SCHEMA = 'elf3_three_rgb_gripper_source/v2'
WRISTS = ('head_rgb', 'left_wrist_rgb', 'right_wrist_rgb')


def enabled(config):
    if config.get('requirements', {}).get('diagnostic_three_rgb_gripper') is not True:
        return False
    if config.get('mode') != 'real' or config['requirements'].get('formal_collection') is not False:
        raise ValueError('wrist diagnostic must be real and explicitly non-formal')
    if config.get('dataset', {}).get('embodiment') != 'elf3_three_rgb_gripper_diagnostic':
        raise ValueError('wrist diagnostic must not claim the native66 training embodiment')
    cameras = config.get('cameras', {})
    if any(cameras.get(n, {}).get('enabled') is not True for n in WRISTS):
        raise ValueError('wrist diagnostic requires both wrist cameras')
    if cameras.get('head_depth', {}).get('enabled') is not False:
        raise ValueError('three-RGB trial explicitly omits depth')
    gripper = config.get('gripper', {})
    if gripper.get('enabled') is not True or gripper.get('mapping', {}).get('verified') is not False:
        raise ValueError('diagnostic records raw gripper coordinates, not a verified mapping')
    expected = validate_identities(gripper.get('expected_identities'))
    if validate_identities(config.get('metadata', {}).get('gripper_identities')) != expected:
        raise ValueError('gripper identity metadata differs from collection configuration')
    if config['requirements'].get('require_pico_calibrated') is not True:
        raise ValueError('diagnostic still requires live calibrated PICO input')
    return True


def convert_raw(value, now, expected=None):
    """Legacy live-receipt validation; buffered collection uses convert_raw_source."""
    state, command = convert_raw_source(value, expected)
    published = float(value['monotonic'])
    if not 0 <= now-published <= .05:
        raise ValueError('stale publication or different monotonic clock')
    for item in value['samples']:
        for stamp in (item['feedback_monotonic'], item['command']['sent_monotonic']):
            if now-float(stamp) > .05:
                raise ValueError('stale feedback/command')
    return state, command


def convert_raw_source(value, expected=None):
    """Validate at publication, independently of transport/callback delay."""
    expected = validate_identities(identities() if expected is None else expected)
    if not isinstance(value, dict):
        raise ValueError('gripper sample must be an object')
    if value.get('schema') != 1 or value.get('training_eligible') is not True:
        raise ValueError('gripper sample is not active human demonstration')
    if value.get('ineligible_reasons'):
        raise ValueError('gripper sample reports ineligible reasons')
    published = float(value['monotonic'])
    if not math.isfinite(published) or published < 0:
        raise ValueError('invalid publication monotonic clock')
    items = value.get('samples', [])
    if (not isinstance(items, list) or len(items) != 2
            or not all(isinstance(x, dict) for x in items)
            or [x.get('side') for x in items] != ['left', 'right']):
        raise ValueError('atomic left/right sample required')
    states, commands = [], []
    for item, identity in zip(items, expected):
        if not isinstance(item.get('command'), dict):
            raise ValueError('gripper command must be an object')
        check_sample_identity(item, identity)
        for stamp in (item['feedback_monotonic'], item['command']['sent_monotonic']):
            if not 0 <= published-float(stamp) <= .05:
                raise ValueError('stale feedback/command at source')
        opened, closed = float(item['open_rad']), float(item['closed_rad'])
        if not all(math.isfinite(v) for v in (opened, closed)) or abs(closed-opened) < .1:
            raise ValueError('invalid soft limits')
        for key, position, result in (
                ('closure_fraction', item['position_rad'], states),
                ('command_closure_fraction', item['command']['position'], commands)):
            position = float(position)
            fraction = (position-opened)/(closed-opened)
            if not math.isfinite(position) or not math.isclose(fraction, float(item[key]), abs_tol=1e-7):
                raise ValueError('inconsistent raw position/closure fraction')
            if not -.01 <= fraction <= 1.01:
                raise ValueError('outside calibrated range')
            result.append(position)
    return np.asarray(states), np.asarray(commands)


def validate_raw_row(gripper, expected=None):
    if gripper.get('coordinate') != 'motor_rad_diagnostic_only':
        raise ValueError('diagnostic gripper coordinate must be raw motor radians')
    if gripper.get('mapping', {}).get('verified') is not False:
        raise ValueError('diagnostic must not claim a verified training mapping')
    raw = gripper['native_raw_sample']
    state, command = convert_raw_source(raw, expected)
    for key, expected in (('state', state), ('command', command)):
        actual = np.asarray(gripper.get(key), dtype=float)
        if actual.shape != (2,) or not np.allclose(actual, expected, rtol=0, atol=1e-7):
            raise ValueError(f'diagnostic {key} differs from raw motor data')
