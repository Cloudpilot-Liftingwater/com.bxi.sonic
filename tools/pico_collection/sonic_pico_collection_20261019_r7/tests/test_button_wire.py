"""Actual producer codec and listener loop; collection child and clocks mocked."""
import importlib.util
import json
from pathlib import Path
import signal
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch
import numpy as np

REPO = Path(__file__).resolve().parents[4]
R7 = REPO/'tools/pico_collection/sonic_pico_collection_20261019_r7'
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(R7), str(R7/'collector')]

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

import button_supervisor as listener
sender = load('actual_producer_codec', REPO/'pico/gear_sonic/utils/teleop/zmq/zmq_planner_sender.py')
from real_robot_collection.pico import PicoSnapshot, decode_packed_message

class WireReplay(unittest.TestCase):
    def test_two_saves_two_discards_each_pause_resume_through_real_loop(self):
        # Arrival occurs AFTER loop entry, BEFORE queue snapshot returns. The
        # production subscriber has an independent receiver thread with this order.
        clock = [100.]
        step = [0]
        requests = []
        for discard in (False, False, True, True):
            requests += [(1,False,False),(1,True,False),(2,False,False),
                         (2,False,False),(1,False,False),(1,not discard,discard),
                         (1,False,False)]
        handlers = {}
        instances = []
        class Subscriber:
            def __init__(self, host, port, topic, **kw):
                self.topic = topic; self.received = 0; self.error = None; self.packet = None
            def start(self): pass
            def close(self): pass
            def latest(self): return self.packet
            def after(self, cursor):
                clock[0] += .002
                mode, toggle, discard = requests[step[0]]
                if self.topic == 'pose' and mode != 1: return ()
                fields = dict(stream_mode=np.array([mode],np.int32),
                              calibration_ready=np.array([True],bool))
                if self.topic == 'pose':
                    fields['timestamp_realtime'] = np.array([clock[0]+1000],np.float64)
                else:
                    fields.update(toggle_data_collection=np.array([toggle],bool),
                                  toggle_data_abort=np.array([discard],bool))
                wire = sender.pack_pose_message(fields, self.topic)
                self.packet = PicoSnapshot(decode_packed_message(wire,self.topic), clock[0])
                self.received += 1
                return (self.packet,)
        def sleep(_):
            clock[0] += .02; step[0] += 1
            if step[0] == len(requests): handlers[signal.SIGTERM]()
        def popen(args, **kwargs):
            folder = Path(args[3])
            child = Mock(pid=123, returncode=0)
            child.poll.side_effect = lambda: 0 if json.loads((folder/'intent.json').read_text())['command'] in ('finish','discard') else None
            return child
        original = listener.Supervisor
        def supervisor(*args, **kwargs):
            instance = original(*args, popen=popen, **kwargs)
            instances.append(instance)
            return instance
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(listener,'Supervisor',side_effect=supervisor), \
             patch.object(listener,'PicoSubscriber',Subscriber), \
             patch.object(listener,'PackedSubscriber',Subscriber), \
             patch.object(listener,'runs_root',return_value=Path(tmp)), \
             patch.object(listener,'require_site'), \
             patch.object(listener.time,'monotonic',side_effect=lambda:clock[0]), \
             patch.object(listener.time,'time',side_effect=lambda:clock[0]+1000), \
             patch.object(listener.time,'sleep',side_effect=sleep), \
             patch.object(listener.signal,'signal',side_effect=lambda sig,fn:handlers.update({sig:fn})), \
             patch.object(sys,'argv',['listener','--left','bxi','--right','bxi']):
            listener.main()
            folders = list(Path(tmp).glob('session_*'))
            self.assertEqual(len(folders),4, 'Fresh button messages must not be rejected as future packets')
            ends = []
            for folder in folders:
                value = json.loads((folder/'intent.json').read_text())
                ends.append(value['command'])
                self.assertEqual(len(value['pauses']),1)
                self.assertIsNotNone(value['pauses'][0][1])
                self.assertLess(value['pauses'][0][0],value['pauses'][0][1])
                self.assertEqual(value['sequence'],4) # start, pause, resume, terminal
            self.assertEqual(sorted(ends),['discard','discard','finish','finish'])

if __name__ == '__main__': unittest.main()
