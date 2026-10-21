"""Collector compatibility with playout-01 rolling SMPL chunks, no live sockets."""
import copy
import hashlib
from pathlib import Path
import unittest

import numpy as np
import test_trial as t
import install_service
from real_robot_collection.pico import PackedSnapshot, smpl_reference_ready, smpl_subscriber


def chunk():
    # _source_chunk_fields from com.bxi.sonic/pico/pose_to_smpl_ref_bridge.py
    return dict(source_chunk=np.array([1], np.uint8),
        source_stream_epoch=np.array([2], np.int64),
        source_received_monotonic_ns=np.array([10_000_000_000], np.int64),
        frame_index=np.arange(1, 11, dtype=np.int64),
        term1_local=np.zeros((10,72), np.float32), root_quat=np.tile([1.,0.,0.,0.], (10,1)),
        wrist=np.zeros((10,6), np.float32), head_joint_pos=np.zeros((10,3), np.float32),
        valid_horizon=np.array([10], np.int32), clamp_slots=np.array([0], np.int32))


class PlayoutInputTests(unittest.TestCase):
    def test_current_chunk_has_no_old_ready_flags_but_is_accepted(self):
        fields = chunk()
        subscriber = smpl_subscriber('127.0.0.1', 5556, 'smpl_ref')
        self.assertTrue(set(subscriber.required_fields) <= fields.keys())
        self.assertNotIn('source_ready', fields)
        self.assertTrue(smpl_reference_ready(PackedSnapshot(fields, 10.)))

    def test_bad_horizon_repeats_nonfinite_or_missing_data_rejected(self):
        for key, value in (
            ('frame_index', np.array([1]*10)), ('frame_index', np.arange(10, dtype=float)),
            ('valid_horizon', np.array([9])), ('clamp_slots', np.array([1])),
            ('term1_local', np.zeros((9,72))), ('root_quat', np.full((10,4), np.nan)),
            ('source_received_monotonic_ns', np.array([0])), ('source_stream_epoch', np.array([0]))):
            fields = chunk(); fields[key] = value
            self.assertFalse(smpl_reference_ready(PackedSnapshot(fields, 10.)), key)
        fields = chunk(); del fields['wrist']
        self.assertFalse(smpl_reference_ready(PackedSnapshot(fields, 10.)))

    def test_legacy_still_requires_ready_calibrated_pose(self):
        snapshot = t.fixture()[-1]
        self.assertTrue(smpl_reference_ready(snapshot))
        for key in ('source_ready', 'source_stream_mode', 'source_calibration_ready'):
            invalid = copy.deepcopy(snapshot); invalid.fields[key] = np.array([0])
            self.assertFalse(smpl_reference_ready(invalid))

    def test_current_r6_service_is_upgradeable_without_changing_gripper_selection(self):
        old = install_service.ROOT.parent/'sonic_pico_collection_20261014_r6'/install_service.NAME
        if not old.exists(): self.skipTest('development baseline only')
        self.assertIn(hashlib.sha256(old.read_bytes()).hexdigest(), install_service.KNOWN_UNITS)
        text = (install_service.ROOT/install_service.NAME).read_text()
        self.assertEqual(text.replace('20261019_r7', '20261014_r6'), old.read_text())


if __name__ == '__main__': unittest.main()
