import ast
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from install_service import install_unit


class InstallTests(unittest.TestCase):
    def test_new_unit_and_idempotent_reapply(self):
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp)/'listener.service'
            self.assertTrue(install_unit(ROOT/'elf3_pico_collection.service', target))
            self.assertFalse(install_unit(ROOT/'elf3_pico_collection.service', target))

    def test_unrelated_unit_or_symlink_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp)/'listener.service'; target.write_text('other')
            with self.assertRaises(RuntimeError): install_unit(ROOT/'elf3_pico_collection.service', target)
            self.assertEqual(target.read_text(), 'other')
            link = Path(temp)/'link.service'; link.symlink_to(target)
            with self.assertRaises(RuntimeError): install_unit(ROOT/'elf3_pico_collection.service', link)

    def test_service_unprivileged_no_force_kill_or_auto_recording(self):
        text = (ROOT/'elf3_pico_collection.service').read_text()
        for line in ('User=bxi', 'KillMode=mixed', 'TimeoutStopSec=infinity', 'SendSIGKILL=no', 'Restart=no'):
            self.assertIn(line, text)
        self.assertNotIn('start_hardware', text)
        self.assertNotIn('ros_elf_launch', text)

    def test_only_button_session_scope_changed(self):
        from build import ALLOWED, ADDED, files, PREVIOUS
        baseline = PREVIOUS
        if not baseline.exists(): self.skipTest('extracted package has no development baseline')
        old, new = files(baseline), files(ROOT)
        self.assertFalse(set(old)-set(new))
        self.assertFalse(set(new)-set(old)-ADDED)
        for name, path in old.items():
            if name not in ALLOWED:
                self.assertEqual(path.read_bytes(),new[name].read_bytes(),name)

    def test_no_controller_deployment_or_motor_publishers_in_new_modules(self):
        for name in ('button_supervisor.py', 'session_node.py', 'install_service.py', 'session_protocol.py'):
            tree = ast.parse((ROOT/name).read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    self.assertNotIn('/canfd_packet/tx', node.value)
                    self.assertNotIn('/motion_commands', node.value)
                    self.assertNotIn('systemctl stop ros_elf', node.value)


if __name__ == '__main__': unittest.main()
