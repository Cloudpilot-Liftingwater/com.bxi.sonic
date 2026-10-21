"""Regression: the actual installed 10/12 unit was rejected by the 10/14-r5 upgrader."""
import contextlib
import hashlib
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import test_demo03 as demo
import install_service as installer
from auto_identity import AutoIdentityGate
from configure import add_gripper_arguments, configuration


class UpgradeTests(unittest.TestCase):
    def test_actual_1012_service_upgrades_with_exact_backup(self):
        payload = (installer.ROOT/'tests/installed_20261012_r5.service').read_bytes()
        self.assertEqual(hashlib.sha256(payload).hexdigest(),
            '31ebf4436009b5139c237704346bf4d1bece28fcbc6752ce6961866bcd192996')
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp)/installer.NAME
            target.write_bytes(payload)
            installer.check_unit(installer.ROOT/installer.NAME,target,replace_known=True)
            self.assertTrue(installer.install_unit(installer.ROOT/installer.NAME,target,replace_known=True))
            backups=list(Path(tmp).glob('elf3_pico_collection_backup_*/*.service'))
            self.assertEqual(len(backups),1)
            self.assertEqual(backups[0].read_bytes(),payload)
            self.assertEqual(target.read_bytes(),(installer.ROOT/installer.NAME).read_bytes())

    def test_active_service_is_never_stopped_by_installer(self):
        with tempfile.TemporaryDirectory() as tmp:
            local=Path(tmp)/'runs'
            def paths(value):
                return local if str(value)=='/home/bxi/pico_collection_runs' else Path(value)
            shown=SimpleNamespace(returncode=0,stderr='',stdout=(
                'LoadState=loaded\nActiveState=active\nDropInPaths=\n'
                'FragmentPath=/etc/systemd/system/elf3_pico_collection.service\n'))
            with mock.patch.object(installer,'Path',side_effect=paths), \
                 mock.patch.object(installer.pwd,'getpwnam',return_value=SimpleNamespace(pw_uid=os.getuid(),pw_gid=os.getgid())), \
                 mock.patch.object(installer.os,'chown'), \
                 mock.patch.object(installer,'locked_file',return_value=contextlib.nullcontext()), \
                 mock.patch.object(installer,'require_idle_status'), \
                 mock.patch.object(installer,'check_unit'), \
                 mock.patch.object(installer.subprocess,'run',return_value=shown) as run:
                with self.assertRaisesRegex(RuntimeError,'listener active; no service stopped'):
                    installer.install_and_enable()
                self.assertEqual([c.args[0][:2] for c in run.call_args_list],[['systemctl','show']])

    def test_stopped_old_stale_or_error_status_never_counts_as_ready(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'status.json'
            good=dict(updated_unix=100.,collector_release=installer.ROOT.name,phase='WAITING_FOR_BUTTON')
            path.write_text(json.dumps(good))
            installer.require_idle_status(path,101.,started_after=99.)
            for change in (dict(collector_release='sonic_pico_collection_20261012_r5'),
                           dict(updated_unix=98.),dict(phase='LISTENER_STOPPED'),
                           dict(phase='ERROR_WAITING_FOR_NEW_BUTTON')):
                path.write_text(json.dumps(dict(good,**change)))
                with self.assertRaises(RuntimeError):
                    installer.require_idle_status(path,101.,started_after=99.)

    def test_auto_mode_retains_current_can_mapping_in_child_contract(self):
        import argparse
        gate=AutoIdentityGate()
        for stamp in (100.,100.02):
            self.assertTrue(gate.accept(json.dumps(demo.demo_sample(stamp)),stamp-90+.01),gate.error)
        argv=gate.arguments(10.03,1)
        parser=argparse.ArgumentParser();add_gripper_arguments(parser)
        args=parser.parse_args(argv)
        cfg=configuration(args.left,args.right,args.left_id,args.right_id,
                          left_bus=args.left_bus,right_bus=args.right_bus)
        self.assertEqual(cfg['gripper']['expected_identities'],
                         demo.selected_config()['gripper']['expected_identities'])
