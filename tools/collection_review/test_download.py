"""Offline selection, transfer-integrity and outcome checks; no robot access."""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


def load(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


plan, review = load("plan"), load("review")


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.base, self.dest = self.root / "remote", self.root / "download"
        self.base.mkdir()
        self.dest.mkdir()
        for i in range(6):
            name = f"session_{i}"
            run = self.base / "pico_collection_runs" / name
            run.mkdir(parents=True)
            discarded = i % 2 == 1
            pauses = [[10., 15.]] if i >= 4 else []
            self.write(run / "intent.json", dict(command="discard" if discarded else "finish",
                                                pauses=pauses))
            self.write(run / "launcher_status.json", dict(exit_code=0, runtime_failure=None))
            episode = self.base / "pico_collection_data" / name / "three_rgb_gripper_source"
            episode /= "_discarded/episode0" if discarded else "episode_0000"
            episode.mkdir(parents=True)
            self.write(episode / "meta.json", dict(
                num_frames=2, discarded=discarded, excluded_from_training=discarded,
                termination={"success": not discarded}, synchronization={"pose_pauses": [[100., 105.]] if pauses else []},
                metadata={"collector_release": review.EXPECTED_RELEASE,
                          "capture_profile": "sonic50_latest_rgb"}))
            (episode / "frames.jsonl").write_text(
                "\n".join(json.dumps({"auxiliary": {"anchor_stamp_unix": x}}) for x in (99., 106.)) + "\n")
            os.utime(run, (100+i, 100+i))

    def write(self, path, value):
        path.write_text(json.dumps(value))

    def copy(self):
        value = plan.build(self.base)
        self.write(self.dest / "manifest.json", value)
        review.prepare(self.dest)
        # Same rsync file-list semantics as the SSH wrapper, using local fixtures.
        subprocess.run(["rsync", "-rt", "--from0", "--files-from=" + str(self.dest / "files.list"),
                        str(self.base) + "/", str(self.dest / "robot") + "/"], check=True)
        return value

    def test_latest_four_includes_failed_session(self):
        (self.base / "pico_collection_runs/session_5/launcher_status.json").unlink()
        value = plan.build(self.base)
        self.assertEqual([s["name"] for s in value["sessions"]],
                         ["session_5", "session_4", "session_3", "session_2"])
        self.assertFalse(value["sessions"][0]["final_report_present"])
        self.assertTrue(any("/_discarded/" in f["path"] for f in value["files"]))

    def test_current_release_selects_every_session_and_excludes_history(self):
        release='20261021-collection-14-official-session'
        self.write(self.base/'pico_collection_runs/status.json',
                   dict(phase='WAITING_FOR_BUTTON',collector_release=release))
        for i in range(6):
            self.write(self.base/f'pico_collection_runs/session_{i}/capture.json',
                       dict(collector_release=release if i else 'old'))
        value=plan.build(self.base,'current-release')
        self.assertEqual([s['name'] for s in value['sessions']],
                         ['session_5','session_4','session_3','session_2','session_1'])
        self.assertEqual(value['selection'],'current-release')

    def test_current_release_requires_known_version(self):
        with self.assertRaisesRegex(RuntimeError,'没有版本号'):
            plan.build(self.base,'current-release')

    def test_all_keeps_saved_discarded_and_orphan_data_across_versions(self):
        orphan = self.base / 'pico_collection_data/session_orphan'
        orphan.mkdir()
        (orphan / 'record.bin').write_bytes(b'preserve me')
        value = plan.build(self.base, 'all')
        self.assertEqual(len(value['sessions']), 7)
        paths = [f['path'] for f in value['files']]
        self.assertEqual(len(paths), len(set(paths)))
        self.assertIn('pico_collection_data/session_orphan/record.bin', paths)
        self.assertTrue(any('/_discarded/' in p for p in paths))
        row = next(s for s in value['sessions'] if s['name'] == 'session_orphan')
        self.assertFalse(row['run_directory_present'])
        self.assertEqual(row['raw_files'], 1)

    def test_all_refuses_symlink_data(self):
        (self.base / 'pico_collection_data/session_escape').symlink_to(self.root)
        with self.assertRaisesRegex(ValueError, 'symlink'):
            plan.build(self.base, 'all')

    def test_free_form_trial_not_required_to_have_two_discards(self):
        # Change one of the four fixture outcomes: three saves and one discard.
        run=self.base/'pico_collection_runs/session_3'
        old=self.base/'pico_collection_data/session_3/three_rgb_gripper_source/_discarded/episode0'
        meta=json.loads((old/'meta.json').read_text())
        meta.update(discarded=False,excluded_from_training=False,termination={'success':True})
        self.write(old/'meta.json',meta)
        self.write(run/'intent.json',dict(command='finish',pauses=[]))
        self.copy()  # Default latest-four must also allow three saves / one discard.
        with patch.dict(sys.modules,verify_saved=SimpleNamespace(verify=lambda *a,**kw: dict(
                readback='PASS',timing_review='PASS',gripper_identities=[
                    dict(side=side,protocol='bxi',bus=bus,motor_id=1)
                    for side,bus in (('left',5),('right',6))])),
                session_protocol=SimpleNamespace(capture_segment=lambda stamp,pauses:int(bool(pauses) and stamp>=105))):
            review.review(self.dest)
        result=json.loads((self.dest/'review.json').read_text())
        self.assertEqual(result['status'],'CURRENT_RELEASE_RAW_CHECK_PASS')
        self.assertEqual(result['categories'],{'SAVED_PASS':3,'DISCARDED_EXCLUDED':1})

    def test_active_selected_session_refused(self):
        self.write(self.base / "pico_collection_runs/status.json",
                   dict(session=str(self.base / "pico_collection_runs/session_5"), phase="AUDITING"))
        with self.assertRaisesRegex(RuntimeError, "仍在"):
            plan.build(self.base)

    def test_symlink_refused(self):
        (self.base / "pico_collection_runs/session_5/outside").symlink_to(self.root)
        with self.assertRaisesRegex(ValueError, "symlink"):
            plan.build(self.base)

    def test_transfer_and_corruption(self):
        value = self.copy()
        review.check_hashes(self.dest, value)
        path = self.dest / "robot" / value["files"][0]["path"]
        content = path.read_bytes()
        path.write_bytes(b"x" + content[1:])
        with self.assertRaisesRegex(ValueError, "SHA256"):
            review.check_hashes(self.dest, value)

    def test_path_escape_refused(self):
        value = plan.build(self.base)
        value["files"][0]["path"] = "../../outside"
        self.write(self.dest / "manifest.json", value)
        with self.assertRaises(ValueError):
            review.prepare(self.dest)

    def test_two_saved_two_discarded_same_episode_pause(self):
        self.copy()
        def verify(path, **kwargs):
            meta = json.loads((path / "meta.json").read_text())
            return dict(readback="PASS", timing_review="INCOMPLETE" if meta["discarded"] else "PASS",
                        uninterrupted_row_rate_hz=50,
                        gripper_identities=[dict(side=side, protocol="bxi", bus=bus, motor_id=1)
                                            for side, bus in (("left", 5), ("right", 6))])
        def segment(stamp, pauses):
            return 0 if not pauses or stamp < 100 else 1 if stamp >= 105 else None
        with patch.dict(sys.modules, verify_saved=SimpleNamespace(verify=verify),
                        session_protocol=SimpleNamespace(capture_segment=segment)):
            review.review(self.dest)
        result = json.loads((self.dest / "review.json").read_text())
        self.assertEqual(result["status"], "CURRENT_RELEASE_RAW_CHECK_PASS")
        self.assertTrue(result['two_save_two_discard_pause_coverage'])
        self.assertEqual(result["categories"], {"SAVED_PASS": 2, "DISCARDED_EXCLUDED": 2})
        self.assertEqual(result["paused_saved"], 1)
        self.assertEqual(result["paused_discarded"], 1)

    def test_running_release_from_manifest_not_workstation_constant(self):
        deployed_release = 'newer-collector-than-review-script'
        self.write(self.base / 'pico_collection_runs/status.json',
                   dict(phase='WAITING_FOR_BUTTON', collector_release=deployed_release))
        for run in (self.base / 'pico_collection_runs').glob('session_*'):
            (run / 'session.log').write_text('COLLECTOR_RELEASE='+deployed_release+'\n')
        for path in (self.base / 'pico_collection_data').rglob('meta.json'):
            meta = json.loads(path.read_text())
            meta['metadata']['collector_release'] = deployed_release
            self.write(path, meta)
        self.copy()
        with patch.dict(sys.modules, verify_saved=SimpleNamespace(verify=lambda *a, **kw: dict(
                readback='PASS', timing_review='PASS', gripper_identities=[
                    dict(side=side, protocol='bxi', bus=bus, motor_id=1)
                    for side, bus in (('left', 5), ('right', 6))])),
                session_protocol=SimpleNamespace(capture_segment=lambda stamp, pauses: int(bool(pauses) and stamp >= 105))):
            review.review(self.dest)
        result = json.loads((self.dest/'review.json').read_text())
        self.assertEqual(result['status'], 'CURRENT_RELEASE_RAW_CHECK_PASS')
        self.assertEqual(result['expected_release'], deployed_release)
        self.assertEqual(result['current_release_sessions'], 4)

    def test_missing_resume_not_claimed_as_pass(self):
        # Commands say resumed, but there are no captured rows after resume.
        episode = self.base / "pico_collection_data/session_4/three_rgb_gripper_source/episode_0000"
        (episode / "frames.jsonl").write_text(json.dumps({"auxiliary": {"anchor_stamp_unix": 99.}}) + "\n")
        self.copy()
        with patch.dict(sys.modules,
                        verify_saved=SimpleNamespace(verify=lambda *a, **kw: dict(
                            readback="PASS", timing_review="PASS", gripper_identities=[])),
                        session_protocol=SimpleNamespace(capture_segment=lambda stamp, _: 0 if stamp < 100 else 1)):
            review.review(self.dest)
        result = json.loads((self.dest / "review.json").read_text())
        row = next(r for r in result["sessions"] if r["session"] == "session_4")
        self.assertFalse(row["pause_resume_verified"])
        self.assertEqual(result["status"], "NEEDS_REVIEW")

    def test_internal_writer_abort_is_counted_even_without_abort_button(self):
        reason = "capture_or_writer_failure:episode writer queue is full; refusing to lose a frame"
        run = self.base / "pico_collection_runs/session_4"
        episode = self.base / "pico_collection_data/session_4/three_rgb_gripper_source/episode_0000"
        meta = json.loads((episode / "meta.json").read_text())
        meta.update(discarded=False, excluded_from_training=True,
                    termination=dict(success=False, reason=reason), synchronization=dict(pose_pauses=[]))
        self.write(episode / "meta.json", meta)
        self.write(run / "session_result.json", dict(outcome="interrupted", reason=reason))
        self.write(run / "launcher_status.json", dict(exit_code=4,
                   runtime_failure=dict(reason="collector_preserved_interrupted_episode")))
        for command in ("start", "finish"):
            with self.subTest(command=command):
                self.write(run / "intent.json", dict(command=command, pauses=[]))
                self.copy()
                with patch.dict(sys.modules, verify_saved=SimpleNamespace(verify=lambda *a, **kw: dict(
                    readback="PASS", timing_review="INCOMPLETE", gripper_identities=[])),
                    session_protocol=SimpleNamespace(capture_segment=lambda *a: 0)):
                    review.review(self.dest)
                result = json.loads((self.dest / "review.json").read_text())
                row = next(r for r in result["sessions"] if r["session"] == "session_4")
                self.assertEqual(result["status"], "FAILED_RECORDING_INTERRUPTED")
                self.assertEqual(result["categories"]["INTERRUPTED_PRESERVED"], 1)
                self.assertEqual(row["failure_reason"], reason)
                self.assertEqual(row["pause_resume_outcome"], "未完成（采集中途故障）")
                self.assertFalse(row["pause_resume_verified"])

    def test_old_logs_not_current_test_results(self):
        shutil.rmtree(self.base / "pico_collection_data")
        for run in (self.base / "pico_collection_runs").glob("session_*"):
            (run / "session.log").write_text(
                "COLLECTOR_RELEASE=sonic_pico_collection_20261014_r6\n"
                "[INFO] [1791977062.645071100] historical test\n")
        self.copy()
        def unexpected_verify(*args, **kwargs):
            raise AssertionError("No episode to verify")
        with patch.dict(sys.modules, verify_saved=SimpleNamespace(verify=unexpected_verify),
                        session_protocol=SimpleNamespace(capture_segment=None)):
            review.review(self.dest)
        result = json.loads((self.dest / "review.json").read_text())
        self.assertEqual(result["status"], "NO_SESSIONS_FROM_CURRENT_RELEASE")
        self.assertTrue(all(s["recorded_at"].startswith("2026-10-14") for s in result["sessions"]))
        self.assertIn("未执行图像或数值读回", (self.dest / "SUMMARY.txt").read_text())


if __name__ == "__main__":
    with contextlib.redirect_stdout(io.StringIO()):
        unittest.main()
