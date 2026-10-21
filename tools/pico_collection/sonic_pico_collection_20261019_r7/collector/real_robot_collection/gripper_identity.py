"""Explicit per-side collection identity, not motor discovery or configuration."""

PROFILES = {'dm': 'dm_follow8', 'bxi': 'bxi_colleague_rad_v1'}
FIELDS = ('side', 'bus', 'motor_id', 'protocol', 'profile', 'position_unit', 'effort_unit')


def identities(left='dm', right='dm', left_id=None, right_id=None, *, left_bus=5, right_bus=6):
    result = []
    for side, bus, protocol, motor in (
            ('left', left_bus, left, left_id), ('right', right_bus, right, right_id)):
        if protocol not in PROFILES:
            raise ValueError('select dm or bxi explicitly; no auto protocol')
        if motor is None:
            motor = 4 if side == 'right' and protocol == 'dm' else 1
        if type(motor) is not int or not 1 <= motor <= 15:
            raise ValueError('collection motor ID must be an integer in 1..15')
        result.append(dict(side=side, bus=bus, motor_id=motor, protocol=protocol,
            profile=PROFILES[protocol], position_unit='rad', effort_unit='Nm'))
    return validate_identities(result)


def validate_identities(value):
    """Return a fresh normalized contract; never learn identity from incoming data."""
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError('explicit left/right gripper identities required')
    for item, side in zip(value, ('left', 'right')):
        if not isinstance(item, dict) or set(item) != set(FIELDS):
            raise ValueError('invalid gripper identity fields')
        if (item['side'] != side or type(item['bus']) is not int or item['bus'] not in (5, 6)
                or type(item['motor_id']) is not int or not 1 <= item['motor_id'] <= 15
                or item['protocol'] not in PROFILES
                or item['profile'] != PROFILES[item['protocol']]
                or item['position_unit'] != 'rad' or item['effort_unit'] != 'Nm'):
            raise ValueError('invalid gripper side/bus/ID/protocol/profile/units')
    if value[0]['bus'] == value[1]['bus']:
        raise ValueError('left/right grippers must use distinct CAN5/CAN6 buses')
    return [{key: item[key] for key in FIELDS} for item in value]


def check_sample_identity(item, expected):
    if (type(item.get('bus')) is not int or type(item.get('motor_id')) is not int
            or any(item.get(key) != expected[key] for key in FIELDS)):
        actual = {key:item.get(key) for key in FIELDS}
        raise ValueError(f'gripper identity mismatch: expected={expected}, received={actual}')
