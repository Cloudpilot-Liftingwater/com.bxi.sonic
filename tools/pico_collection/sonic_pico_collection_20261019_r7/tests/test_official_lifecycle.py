import sys
from pathlib import Path
import unittest
import numpy as np
R=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(R),str(R/'collector'),str(R/'tests')]
from test_button_recording import OfficialButtons,manager
from real_robot_collection.pico import PicoSnapshot
from session_protocol import ButtonGate,read_intent

class NoAliases(OfficialButtons):
    def test_old_ya_yb_flags_cannot_start_or_save_even_in_manager_packet(self):
        self.event(manager(1))
        def old(t,start=False,finish=False):
            p=manager(t)
            p.fields.update(start_data_collection=np.array([start]),finish_data_collection=np.array([finish]))
            return p
        self.event(old(1.1,start=True));self.popen.assert_not_called()
        self.event(manager(1.2));self.event(manager(1.3,toggle=True))
        self.assertEqual(read_intent(self.s.folder)['command'],'start')
        self.event(old(1.4,finish=True))
        self.assertEqual(read_intent(self.s.folder)['command'],'start')
        self.event(manager(1.5,toggle=True))
        self.assertEqual(read_intent(self.s.folder)['command'],'finish')

    def test_official_state_does_not_reset_on_ax_pause_or_resume(self):
        self.event(manager(1));self.event(manager(1.1,toggle=True))
        state=self.s.episode_state
        self.assertEqual(state.get_state(),state.RECORDING)
        self.event(manager(1.2,mode=2),pose=False)
        self.assertEqual(state.get_state(),state.RECORDING)
        self.event(manager(1.3));self.assertEqual(state.get_state(),state.RECORDING)
        self.event(manager(1.4,toggle=True))
        self.assertEqual(state.get_state(),state.NEED_TO_SAVE)
        self.child.returncode=0;self.child.poll.return_value=0;self.s.poll()
        self.assertEqual(state.get_state(),state.IDLE)

if __name__=='__main__':
 suite=unittest.TestSuite(NoAliases(n) for n in ('test_old_ya_yb_flags_cannot_start_or_save_even_in_manager_packet','test_official_state_does_not_reset_on_ax_pause_or_resume'))
 r=unittest.TextTestRunner().run(suite)
 raise SystemExit(not r.wasSuccessful())
