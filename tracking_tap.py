"""Optional, lossy observation only. Never changes policy arrays or motor state.

An abstract Unix datagram socket stays on this Linux host. No recorder -> no
serialization; failed/slow recorder -> drop. No filesystem I/O, thread, blocking
send, ROS publication, or model replay on the control path.
"""
import json
import os
import socket
import struct
import time
import uuid

import numpy as np

ADDRESS = '\0elf3_sonic_tracking_v1'
MAGIC = b'STK1'
ARRAYS = ('model_input', 'raw_action', 'clipped_action', 'decoded_target',
          'policy_target', 'observed_q', 'observed_dq', 'motion_token', 'root_quat')
SIZES = (1770, 29, 29, 29, 29, 29, 29, 64, 40)
_socket = None
_retry_at = 0.
_session = uuid.uuid4().hex
_sent = _dropped = _errors = 0
_max_overhead_ns = 0


def capture(policy, reference, source, model_input, raw, clipped, decoded, q, dq, started):
    """Exception-contained snapshot; called after the actual backend run."""
    global _socket, _retry_at, _sent, _dropped, _errors, _max_overhead_ns
    begin = time.monotonic_ns()
    try:
        now = begin * 1e-9
        if _socket is None:
            if now < _retry_at:
                return
            _retry_at = now + 1.
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
            sock.setblocking(False)
            try:
                sock.connect(ADDRESS)
            except OSError:
                sock.close()
                return
            _socket = sock
        metadata = dict(schema=1, session=_session, pid=os.getpid(),
            sequence=int(policy.inference_sequence), source=source,
            policy_status=policy.last_status, source_blend_active=bool(policy.source_blend_active),
            reference_frame_index=int(reference.frame_index), reference_sequence=int(reference.sequence),
            reference_received_monotonic=float(policy.latest_live_ref_time),
            started_monotonic=float(started), completed_monotonic=now,
            completed_wall_ns=time.time_ns(), model_id=policy.model_id,
            model_sha256=policy.model_sha256, joint_names=list(policy.joint_names),
            default_q=np.asarray(policy.default_dof_pos).tolist(),
            action_scale=np.asarray(policy.action_scale).tolist(),
            yaw_offset_rad=float(policy.yaw_offset),
            tap_sent=_sent, tap_dropped=_dropped, tap_errors=_errors,
            tap_previous_max_overhead_ns=_max_overhead_ns)
        arrays = (model_input, raw, clipped, decoded, policy.target_dof_pos,
                  q, dq, policy.motion_token, reference.root_quat)
        if any(np.asarray(a).size != n for a, n in zip(arrays, SIZES)):
            raise ValueError('tracking array size mismatch')
        header = json.dumps(metadata, separators=(',', ':'), allow_nan=False).encode()
        body = b''.join(np.asarray(a, dtype='<f4').tobytes(order='C') for a in arrays)
        packet = MAGIC + struct.pack('<I', len(header)) + header + body
        _socket.send(packet)
        _sent += 1
    except (BlockingIOError, InterruptedError):
        _dropped += 1
    except OSError:
        _dropped += 1
        if _socket is not None:
            _socket.close()
            _socket = None
        _retry_at = time.monotonic() + 1.
    except Exception:
        _errors += 1
    finally:
        _max_overhead_ns = max(_max_overhead_ns, time.monotonic_ns()-begin)


def decode(packet):
    if packet[:4] != MAGIC or len(packet) < 8:
        raise ValueError('unknown trace packet')
    length, = struct.unpack('<I', packet[4:8])
    if length > 16384 or len(packet) != 8+length+4*sum(SIZES):
        raise ValueError('trace packet size mismatch')
    meta = json.loads(packet[8:8+length])
    if meta.get('schema') != 1:
        raise ValueError('unknown trace schema')
    offset = 8+length
    result = {}
    for name, size in zip(ARRAYS, SIZES):
        array = np.frombuffer(packet, dtype='<f4', count=size, offset=offset).copy()
        if not np.isfinite(array).all():
            raise ValueError('nonfinite '+name)
        result[name] = array
        offset += 4*size
    return meta, result
