"""Byte-preserving optional observation hook for reviewed policy sources."""
import ast
IMPORT = '''\n# Optional observation only; unavailable recorder never blocks control.
try:
    from .tracking_tap import capture as _capture_tracking
except Exception:
    _capture_tracking = None
'''
HOOK = '''        if _capture_tracking is not None:
            try:
                _capture_tracking(self, frame, source, model_input, raw_action,
                                  action, candidate, q, dq, now_mono)
            except Exception:
                pass  # Observation failures must never affect the robot.
'''

def patched(source):
    head = '\nimport zmq\n'
    if source.count(head) != 1 or '_capture_tracking' in source:
        raise ValueError('policy hook anchors do not match')
    tree = ast.parse(source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'SonicTeleopPolicy')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'inference_step')
    tail = method.body[-1]
    if not isinstance(tail, ast.Return) or ast.unparse(tail.value) != 'self.target_dof_pos':
        raise ValueError('policy return anchor does not match')
    lines = source.splitlines(keepends=True)
    lines[tail.lineno-1:tail.lineno-1] = [HOOK]
    result = ''.join(lines).replace(head, head+IMPORT, 1)
    compile(result, 'policy.py', 'exec')
    return result
