"""Exercise transfer orchestration with fake SSH and actual manifest validation."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import download


class TransferTests(unittest.TestCase):
    def test_all_download_keeps_trusted_ssh_and_verifies_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = Path(tmp)/'report'
            raw = b'saved test payload'
            path = 'pico_collection_data/session_one/episode_0000/data.bin'
            value = dict(schema='elf3-collection-download-v1', remote_base='/custom/data root',
                         sessions=[dict(name='session_one')],
                         files=[dict(path=path, size=len(raw), sha256=hashlib.sha256(raw).hexdigest())])
            calls = []

            def execute(argv, **kw):
                calls.append(argv)
                if argv[0] == 'ssh' and argv[-1].startswith('python3'):
                    kw['stdout'].write(json.dumps(value).encode())
                if argv[0] == 'rsync':
                    dest = report/'robot'/path
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(raw)

            args = argparse.Namespace(host='robot@unit.local', host_key_alias=None,
                                      resume=None, output=report, all=True, remote_base='~/data root')
            with patch.object(download.socket, 'getaddrinfo', return_value=[(2, 1, 6, '', ('192.0.2.1', 22))]), \
                    patch.object(download.subprocess, 'run', side_effect=execute):
                download.run(args)
            self.assertEqual(calls[0][-1], 'true')
            self.assertIn('StrictHostKeyChecking=yes', calls[0])
            self.assertIn('HostKeyAlias=unit.local', calls[0])
            self.assertIn('--all', calls[1][-1])
            self.assertIn('robot@192.0.2.1:/custom/data root/', calls[2])
            self.assertEqual((report/'robot'/path).read_bytes(), raw)

    def test_invalid_target_rejected_before_network(self):
        args = argparse.Namespace(host='-oProxyCommand=bad')
        with patch.object(download.socket, 'getaddrinfo') as dns:
            with self.assertRaises(ValueError):
                download.run(args)
            dns.assert_not_called()

    def test_resume_refuses_different_host_key_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = Path(tmp)
            (report/'manifest.json').write_text(json.dumps(dict(schema='elf3-collection-download-v1', sessions=[], files=[])))
            (report/'connection.json').write_text(json.dumps(dict(user='robot', host_key_alias='original.local')))
            args = argparse.Namespace(host='robot@another.local', host_key_alias=None,
                                      resume=report, output=None, all=False, remote_base='~')
            with patch.object(download.socket, 'getaddrinfo', return_value=[(2, 1, 6, '', ('192.0.2.1', 22))]), \
                    patch.object(download.subprocess, 'run') as execute:
                with self.assertRaisesRegex(ValueError, 'original SSH'):
                    download.run(args)
                execute.assert_not_called()
