"""Synthetic demo-03 producer sample -> ROS callback -> JPEG/JSON -> readback."""
import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock

import test_trial as t
from configure import add_gripper_arguments, configuration
from real_robot_collection.gripper_identity import identities, validate_identities
from real_robot_collection.wrist_diagnostic import convert_raw_source
from real_robot_collection.synchronization import JsonSample


def demo_sample(stamp=100.):
    raw = json.loads((t.ROOT/'tests/demo03_gripper_sample.json').read_text())['sample']
    offset = stamp-100.
    for key in ('monotonic', 'control_monotonic'):
        raw[key] += offset
    for item in raw['samples']:
        item['feedback_monotonic'] += offset
        item['trigger_monotonic'] += offset
        item['command']['sent_monotonic'] += offset
    ns = round((stamp-.005)*1e9)
    raw['stamp'] = dict(sec=ns//10**9, nanosec=ns%10**9)
    return raw


def selected_config():
    parser = argparse.ArgumentParser()
    add_gripper_arguments(parser)
    unit = (t.ROOT/'elf3_pico_collection.service').read_text()
    args = parser.parse_args(unit.split('/listen.sh ', 1)[1].splitlines()[0].split())
    return configuration(args.left, args.right, args.left_id, args.right_id,
                         left_bus=args.left_bus, right_bus=args.right_bus)


class Demo03Tests(unittest.TestCase):
    def test_old_dm_and_old_bxi_mapping_reject_current_sample(self):
        for expected in (identities(), identities('bxi', 'bxi')):
            with self.assertRaisesRegex(ValueError, 'identity mismatch'):
                convert_raw_source(demo_sample(), expected)

    def test_release_service_selects_physical_left_can6_right_can5(self):
        cfg = selected_config()
        expected = cfg['gripper']['expected_identities']
        self.assertEqual([(s['protocol'], s['bus'], s['motor_id']) for s in expected],
                         [('bxi', 6, 1), ('bxi', 5, 1)])
        state, command = convert_raw_source(demo_sample(), expected)
        t.np.testing.assert_allclose(t.np.degrees(state), [60, 240])
        t.np.testing.assert_allclose(t.np.degrees(command), [55, 245])
        self.assertEqual(cfg['metadata']['gripper_identities'], expected)

    def test_duplicate_or_unselected_bus_still_rejected(self):
        for bus in (6, 0, 7, True):
            expected = selected_config()['gripper']['expected_identities']
            expected[1]['bus'] = bus
            with self.assertRaises(ValueError):
                validate_identities(expected)

    def test_current_sample_records_100_frames_300_jpegs_and_reads_back(self):
        mod = t.load_node()
        with tempfile.TemporaryDirectory() as temp:
            cfg = selected_config()
            cfg['dataset']['output_root'] = temp
            # Legacy source-clock fixture, independent of the current site default.
            cfg['metadata']['capture_profile'] = 'head30'
            cfg['metadata']['storage_backend'] = 'legacy_jpeg'
            cfg['metadata']['sample_selection'] = 'legacy_source_match'
            cfg['dataset']['fps'] = 30
            config_path = Path(temp)/'config.yaml'
            config_path.write_text(t.yaml.safe_dump(cfg))
            with mock.patch('real_robot_collection.node.PicoSubscriber'), \
                    mock.patch('real_robot_collection.node.smpl_subscriber'):
                node = mod.SourceCollectionNode(config_path, 'SYNTHETIC_DEMO03_ONLY')
            try:
                node.clock_offset = 90.
                node.recorder.start_episode(dict(task_id='TEST', instruction_text='SYNTHETIC ONLY'), runtime_metadata={})
                for mount in node.recorder.meta['camera_contract']['mounts'].values():
                    mount.update(width=32, height=24)
                for index in range(100):
                    stamp = 100+index/30
                    raw = demo_sample(stamp)
                    with mock.patch.object(mod.time, 'monotonic', return_value=stamp-90+.08):
                        node._on_native_gripper(types.SimpleNamespace(data=json.dumps(raw)))
                    self.assertEqual(len(node.gripper_states), index+1)
                    fixture = t.fixture(index=index, stamp=stamp)
                    fixture[1]['gripper'] = [node.gripper_states[-1]]
                    bundle, missing = t.select(fixture)
                    self.assertEqual(missing, [])
                    bundle['rgb'] = {name:replace(img, data=t.np.zeros((24,32,3), t.np.uint8))
                                     for name,img in bundle['rgb'].items()}
                    node._save_bundle(fixture[0], stamp-90, bundle,
                        JsonSample(stamp-90-.1, stamp-.1, {'current':{'name':node._required_state_name()}}))
                    node.recorder.queue.join()
                node.recorder.meta['synchronization'] = node.sync_summary()
                episode = Path(node.recorder.end_episode({'success':True},100)['path'])
                result = t.verify(episode)
                self.assertEqual((result['readback'], result['timing_review']), ('PASS', 'PASS'))
                self.assertEqual((result['saved_rows'], result['decoded_images']), (100, 300))
                row = json.loads((episode/'frames.jsonl').read_text().splitlines()[0])
                t.np.testing.assert_allclose(t.np.degrees(row['gripper']['state']), [60,240])
                self.assertEqual([s['trigger'] for s in row['gripper']['native_raw_sample']['samples']], [.8,.2])
                self.assertFalse(result['training_export_allowed'])
            finally:
                node.close()
