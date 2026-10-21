"""Synthetic/offline only: DM/BXI config, ingress, persistence, and strict readback."""
import argparse
import copy
import importlib.util
import itertools
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock

import test_trial as t
from configure import add_gripper_arguments, configuration
from check_config import validate_config
from button_supervisor import Supervisor
from real_robot_collection.gripper_identity import identities, validate_identities
from real_robot_collection.gripper_history import GripperHistory
from real_robot_collection.wrist_diagnostic import convert_raw_source, validate_raw_row
from real_robot_collection.synchronization import JsonSample
from verify_saved import verify


def fixture_pair(left='dm', right='dm', **kw):
    f = t.fixture(**kw)
    for item, identity in zip(f[1]['gripper'][0].raw['samples'], identities(left, right)):
        item.update(identity)
    return f


class IdentityTests(unittest.TestCase):
    def test_default_current_robot_dm_ids_and_units(self):
        expected = identities()
        self.assertEqual([(s['protocol'],s['bus'],s['motor_id']) for s in expected],
                         [('dm',5,1),('dm',6,4)])
        self.assertTrue(all(s['profile']=='dm_follow8' and s['position_unit']=='rad'
                            and s['effort_unit']=='Nm' for s in expected))

    def test_all_four_explicit_combinations_config_and_ingest(self):
        for left, right in itertools.product(('dm','bxi'), repeat=2):
            with self.subTest(left=left,right=right):
                cfg = configuration(left,right)
                validate_config(cfg,formal=False,allow_placeholders=False)
                expected = cfg['gripper']['expected_identities']
                raw = fixture_pair(left,right)[1]['gripper'][0].raw
                samples, reason = GripperHistory(expected=expected).ingest(json.dumps(raw),10.08,90.)
                self.assertIsNone(reason)
                self.assertEqual(samples[0].values.tolist(),[2.5,2.5])
                self.assertEqual(cfg['metadata']['gripper_identities'],expected)

    def test_other_protocol_is_not_auto_accepted(self):
        raw = fixture_pair('bxi','bxi')[1]['gripper'][0].raw
        with self.assertRaisesRegex(ValueError,'identity mismatch'):
            convert_raw_source(raw,identities())
        raw = fixture_pair()[1]['gripper'][0].raw
        with self.assertRaisesRegex(ValueError,'identity mismatch'):
            convert_raw_source(raw,identities('bxi','bxi'))

    def test_wrong_side_bus_id_profile_units_and_type_rejected(self):
        mutations = [('side','left'),('bus',5),('bus',6.),('motor_id',1),('motor_id',True),
                     ('protocol','bxi'),('profile','bxi_colleague_rad_v1'),
                     ('position_unit','mm'),('effort_unit','N')]
        for key,value in mutations:
            with self.subTest(key=key,value=value):
                raw = fixture_pair()[1]['gripper'][0].raw
                raw['samples'][1][key] = value
                with self.assertRaises(ValueError):convert_raw_source(raw,identities())
        for key in ('position_unit','effort_unit','profile','motor_id'):
            raw = fixture_pair()[1]['gripper'][0].raw
            del raw['samples'][0][key]
            with self.assertRaises(ValueError):convert_raw_source(raw,identities())

    def test_explicit_nondefault_id_is_pinned(self):
        selected = identities('dm','dm',2,7)
        raw = fixture_pair()[1]['gripper'][0].raw
        with self.assertRaises(ValueError):convert_raw_source(raw,selected)
        raw['samples'][0]['motor_id']=2; raw['samples'][1]['motor_id']=7
        convert_raw_source(raw,selected)
        with self.assertRaises(ValueError):convert_raw_source(raw,identities())

    def test_invalid_config_is_not_a_bypass(self):
        for value in (None,[],{},identities()[::-1]):
            with self.assertRaises(ValueError):validate_identities(value)
        for key,value in (('profile','arbitrary'),('bus',True),('motor_id',0),
                          ('motor_id',16),('position_unit','mm')):
            bad = identities();bad[0][key]=value
            with self.assertRaises(ValueError):validate_identities(bad)
        for protocol in ('auto','DM','other'):
            with self.assertRaises(ValueError):identities(protocol,'dm')
        for value in (True,0,16,'4'):
            with self.assertRaises(ValueError):identities(right_id=value)
        cfg=configuration();cfg['metadata']['gripper_identities']=identities('bxi','bxi')
        with self.assertRaises(ValueError):validate_config(cfg,formal=False,allow_placeholders=False)
        cfg=configuration();del cfg['gripper']['expected_identities']
        with self.assertRaises(ValueError):validate_config(cfg,formal=False,allow_placeholders=False)

    def test_protocol_change_blocks_fallback_until_valid_recovery(self):
        expected=identities(); history=GripperHistory(expected=expected)
        expected[1]['motor_id']=9  # Constructor takes a value snapshot, not mutable config.
        good=fixture_pair()[1]['gripper'][0].raw
        self.assertIsNotNone(history.ingest(json.dumps(good),10.08,90.)[0])
        wrong=fixture_pair('dm','bxi',stamp=100.02)[1]['gripper'][0].raw
        self.assertIsNone(history.ingest(json.dumps(wrong),10.09,90.)[0])
        self.assertEqual(len(history.invalid_intervals),1)
        recovered=fixture_pair(stamp=100.04)[1]['gripper'][0].raw
        self.assertIsNotNone(history.ingest(json.dumps(recovered),10.10,90.)[0])

    def test_eligibility_freshness_and_soft_range_still_enforced(self):
        for kind in ('inactive','stale','range','fraction','zero_span','nan'):
            raw=fixture_pair()[1]['gripper'][0].raw
            if kind=='inactive':raw['training_eligible']=False
            elif kind=='stale':raw['samples'][1]['command']['sent_monotonic']=9.8
            elif kind=='range':raw['samples'][1].update(position_rad=-1,closure_fraction=1.2)
            elif kind=='fraction':raw['samples'][1]['closure_fraction']=.1
            elif kind=='zero_span':raw['samples'][1]['open_rad']=0
            else:raw['samples'][1]['position_rad']=float('nan')
            with self.subTest(kind=kind),self.assertRaises(ValueError):convert_raw_source(raw,identities())

    def test_both_motor_angle_directions_remain_raw(self):
        raw=fixture_pair()[1]['gripper'][0].raw
        raw['samples'][1].update(open_rad=-5.,closed_rad=0.,position_rad=-2.5)
        raw['samples'][1]['command']['position']=-1.
        state,command=convert_raw_source(raw,identities())
        self.assertEqual(state.tolist(),[2.5,-2.5])
        self.assertEqual(command.tolist(),[1.,-1.])

    def test_cli_defaults_and_selection(self):
        parser=argparse.ArgumentParser();add_gripper_arguments(parser)
        args=parser.parse_args([])
        self.assertEqual((args.left,args.right,args.left_id,args.right_id),('dm','dm',None,None))
        args=parser.parse_args(['--left','bxi','--right','dm','--right-id','4'])
        self.assertEqual(configuration(args.left,args.right,args.left_id,args.right_id)
                         ['gripper']['expected_identities'],identities('bxi','dm'))

    def test_pico_supervisor_passes_selected_identities_to_child(self):
        with tempfile.TemporaryDirectory() as temp:
            child=mock.Mock(poll=mock.Mock(return_value=None))
            popen=mock.Mock(return_value=child)
            args=('--left','dm','--right','bxi','--left-id','1','--right-id','1')
            supervisor=Supervisor(temp,Path(temp)/'status.json',popen,collection_args=args)
            try:
                supervisor.event('start')
                self.assertEqual(tuple(popen.call_args.args[0][-8:]),args)
            finally:
                if supervisor.log:supervisor.log.close()

    def test_actual_dual10_telemetry_matches_collector_contract(self):
        path=t.ROOT.parent/'sonic_dual_gripper_20260929_r10/com.bxi.sonic/collection_telemetry.py'
        if not path.is_file():self.skipTest('development controller source not in standalone package')
        spec=importlib.util.spec_from_file_location('read_only_telemetry',path)
        producer=importlib.util.module_from_spec(spec);spec.loader.exec_module(producer)
        NS=types.SimpleNamespace
        for left,right in itertools.product(('dm','bxi'),repeat=2):
            expected=identities(left,right)
            state=NS(_gripper_session_active=True,_gripper_armed=True,_gripper_calibrated=True,
                _gripper_motion_ready=True,_gripper_faulted=False,_left_bus=5,_right_bus=6,
                _gripper_profiles={s['bus']:NS(can_id=s['motor_id'],protocol=NS(value=s['protocol'])) for s in expected},
                _soft_limits_for=lambda *_:(5.,0.),_gripper_exit_started_at=None,
                _gripper_pose_paused=False,_gripper_trigger_timeout_s=.2,policy=NS(last_status='live_reference'))
            state._gripper_control_snapshot=dict(monotonic=9.994,
                feedback={bus:NS(received_at=9.99,position=2.5,velocity=0.,torque=0.) for bus in (5,6)},
                inputs={side:(.8,9.99,None) for side in ('left','right')},
                sides={bus:dict(active=True,mode='tracking') for bus in (5,6)},
                exiting=False,pose_allowed=True,policy_status='live_reference')
            commands={bus:dict(position=1.,sent_monotonic=9.996) for bus in (5,6)}
            raw=producer.gripper_snapshot(state,commands,10.)
            self.assertIsNotNone(raw)
            self.assertTrue(raw['training_eligible'])
            convert_raw_source(raw,expected)


class PersistenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.mod=t.load_node()

    def test_configured_identity_reaches_callback_writer_and_readback_all_pairs(self):
        for left,right in itertools.product(('dm','bxi'),repeat=2):
            with self.subTest(left=left,right=right),tempfile.TemporaryDirectory() as temp:
                cfg=configuration(left,right);cfg['dataset']['output_root']=temp
                # Legacy source-clock fixture, independent of the current site default.
                cfg['metadata']['capture_profile'] = 'head30'
                cfg['metadata']['storage_backend'] = 'legacy_jpeg'
                cfg['metadata']['sample_selection'] = 'legacy_source_match'
                cfg['dataset']['fps'] = 30
                path=Path(temp)/'trial.yaml';path.write_text(t.yaml.safe_dump(cfg))
                with mock.patch('real_robot_collection.node.PicoSubscriber'),mock.patch('real_robot_collection.node.smpl_subscriber'):
                    n=self.mod.SourceCollectionNode(path,'SYNTHETIC_IDENTITY_TEST_ONLY')
                try:
                    n.clock_offset=90.
                    f=fixture_pair(left,right)
                    with mock.patch.object(self.mod.time,'monotonic',return_value=10.08):
                        n._on_native_gripper(types.SimpleNamespace(data=json.dumps(f[1]['gripper'][0].raw)))
                    self.assertEqual(len(n.gripper_states),1)
                    n.recorder.start_episode(dict(task_id='TEST',instruction_text='SYNTHETIC ONLY'),runtime_metadata={})
                    # Small images keep the full writer/decoder test light; runtime camera config is unchanged.
                    for mount in n.recorder.meta['camera_contract']['mounts'].values():
                        mount.update(width=32,height=24)
                    for index in range(100):
                        stamp=100+index/30
                        f=fixture_pair(left,right,index=index,stamp=stamp)
                        bundle,missing=t.select(f);self.assertEqual(missing,[])
                        from dataclasses import replace
                        bundle['rgb']={name:replace(img,data=t.np.zeros((24,32,3),t.np.uint8))
                                       for name,img in bundle['rgb'].items()}
                        n._save_bundle(f[0],stamp-90,bundle,JsonSample(stamp-90-.1,stamp-.1,
                            {'current':{'name':'com.bxi.sonic/sonic_teleop'}}))
                        n.recorder.queue.join()
                    n.recorder.meta['synchronization']=n.sync_summary()
                    episode=Path(n.recorder.end_episode({'success':True},100)['path'])
                    result=verify(episode)
                    self.assertEqual(result['readback'],'PASS')
                    self.assertEqual(result['timing_review'],'PASS')
                    self.assertEqual(result['gripper_identities'],identities(left,right))
                    self.assertEqual(result['decoded_images'],300)
                    meta=json.loads((episode/'meta.json').read_text())
                    self.assertEqual(meta['schema_version'],'elf3_three_rgb_gripper_source/v2')
                    self.assertFalse(meta['training_export_allowed'])
                    frames=(episode/'frames.jsonl').read_text().splitlines()
                    bad=json.loads(frames[0]);bad['gripper']['native_raw_sample']['samples'][1]['motor_id']=9
                    (episode/'frames.jsonl').write_text('\n'.join([json.dumps(bad),*frames[1:]])+'\n')
                    with self.assertRaisesRegex(ValueError,'identity mismatch'):verify(episode)
                    (episode/'frames.jsonl').write_text('\n'.join(frames)+'\n')
                    del meta['gripper_identity_contract']
                    (episode/'meta.json').write_text(json.dumps(meta))
                    with self.assertRaisesRegex(ValueError,'identities required'):verify(episode)
                finally:n.close()


if __name__=='__main__':unittest.main()
