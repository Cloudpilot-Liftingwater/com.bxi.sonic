"""Official latest-value semantics through the real ROS adapter/writer; fake IO only."""
from dataclasses import replace
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import yaml
import test_latest_rgb as rgb
from real_robot_collection.source_sync import latest_received_bundle, AnchorBuffer
import trial_lifecycle as life


class LatestReceivedTests(unittest.TestCase):
    def test_old_and_later_state_are_kept_and_no_command_equality_join(self):
        for offset in (-.031, -.2, .005):
            a,h,images,p,s = rgb.source50()
            h['state'] = [replace(h['state'][0],stamp_unix=a.stamp_unix+offset)]
            h['imu'] = [replace(h['imu'][0],stamp_unix=a.stamp_unix+offset)]
            h['command'] = [replace(h['command'][0],stamp_unix=a.stamp_unix+offset,
                                    position=np.ones(31))]
            b,missing = latest_received_bundle(a,h,images,[p],[s],90.)
            self.assertFalse(missing)
            self.assertIs(b['state'],h['state'][0])
            self.assertIs(b['command'],h['command'][0])
            self.assertEqual(b['smpl_stamp'],s.received_monotonic+90.)

    def test_initial_missing_waits_and_does_not_invent_state(self):
        a,h,images,p,s=rgb.source50()
        h['imu']=[]
        b,missing=latest_received_bundle(a,h,images,[p],[s],90.)
        self.assertEqual(missing,['imu'])
        self.assertIsNone(b['imu'])

    def test_latest_received_chosen_even_if_source_clock_older(self):
        a,h,images,p,s=rgb.source50()
        recent=replace(h['state'][0],received_monotonic=10.1,stamp_unix=99.8)
        h['state']=[recent,h['state'][0]]
        b,missing=latest_received_bundle(a,h,images,[p],[s],90.)
        self.assertFalse(missing)
        self.assertIs(b['state'],recent)

    def test_no_artificial_inference_when_no_new_result(self):
        a,h,*_=rgb.source50()
        buffer=AnchorBuffer(wait_s=0,max_age_s=float('inf'))
        actual=buffer.ready_inference(h['inference'],100.001)
        self.assertIsNotNone(actual)
        buffer.consume(actual,100.001)
        self.assertIsNone(buffer.ready_inference(h['inference'],101.))

    def test_delayed_numerics_survive_real_capture_and_workstation_readback(self):
        mod=rgb.t.load_node()
        with tempfile.TemporaryDirectory() as temp:
            cfg=rgb.config50();cfg['dataset']['output_root']=temp
            path=Path(temp)/'trial.yaml';path.write_text(yaml.safe_dump(cfg))
            with patch('real_robot_collection.node.PicoSubscriber'),patch('real_robot_collection.node.smpl_subscriber'):
                n=mod.SourceCollectionNode(path,'SYNTHETIC_LATEST_ONLY')
            try:
                n.recorder.start_episode(dict(task_id='TEST',instruction_text='SYNTHETIC ONLY'),runtime_metadata={})
                n.source_buffer.reset(100.);n.clock_offset=90.
                # Separate reception: ROS KEEP_LAST=1 equivalent to upstream CONFLATE;
                # inference events retained in order, not conflated into repeated tokens.
                for key in ('state','command','imu'):
                    topic=cfg['robot'][{'state':'actuator_state_topic','command':'actuator_command_topic','imu':'imu_topic'}[key]]
                    qos,opts=n.subscription_options[topic]
                    self.assertEqual(qos.depth,1)
                    self.assertIs(opts['callback_group'],n.input_callback_groups[key])
                self.assertEqual(len({id(g) for g in n.input_callback_groups.values()}),5)
                for i in range(110):
                    a,h,images,p,s=rgb.source50(i)
                    for key in ('state','command','imu'):
                        h[key]=[replace(h[key][0],stamp_unix=a.stamp_unix-.08)]
                    h['command']=[replace(h['command'][0],position=np.ones(31))]
                    for attr,key in (('states','state'),('commands','command'),('imus','imu'),('gripper_states','gripper'),('sonic_inferences','inference')):
                        getattr(n,attr).extend(h[key])
                    for name,values in images.items():n.images[name].extend(values)
                    n.pico.after.return_value=[p];n.smpl.after.return_value=[s]
                    n.status_history.append(rgb.t.JsonSample(a.stamp_unix-90-.01,a.stamp_unix-.01,{'current':{'name':n._required_state_name()}}))
                    if i==0:
                        imu=n.imus.pop()
                        with patch.object(mod.time,'time',return_value=a.stamp_unix+.26),patch.object(mod.time,'monotonic',return_value=a.stamp_unix-90+.26):
                            n._capture_tick()
                        self.assertEqual(n.source_buffer.last_sequence,-1)
                        n.imus.append(imu)
                    with patch.object(mod.time,'time',return_value=a.stamp_unix+.26),patch.object(mod.time,'monotonic',return_value=a.stamp_unix-90+.26):
                        n._capture_tick()
                    n.recorder.queue.join()
                self.assertEqual(n.recorder.frame_count,110)
                self.assertFalse(n.sync_rejections)
                n.recorder.meta['synchronization']=n.sync_summary()
                episode=Path(n.recorder.end_episode({'success':True},100)['path'])
                with patch.object(life,'verify',side_effect=AssertionError('JPEG audit on robot forbidden')),contextlib.redirect_stdout(io.StringIO()):
                    report=life.audit_after_stop(episode,episode.parent,None,full_readback=False)
                self.assertEqual(report['exit_code'],0)
                self.assertEqual(report['saved_data_checks'][0]['decoded_images'],0)
                audit=rgb.verify(episode)
                self.assertEqual(audit['decoded_images'],330)
                self.assertEqual(audit['readback'],'PASS')
                self.assertAlmostEqual(audit['uninterrupted_row_rate_hz'],50.)
                import os,shutil
                if os.environ.get('LATEST_RECEIVED_FIXTURE'):
                    shutil.copytree(episode,os.environ['LATEST_RECEIVED_FIXTURE'])
            finally:n.close()

    def test_writer_failure_still_fails_fast_without_image_decode(self):
        with tempfile.TemporaryDirectory() as temp,contextlib.redirect_stdout(io.StringIO()):
            root=Path(temp);episode=root/'episode_0000';episode.mkdir()
            (episode/'frames.jsonl').write_text('')
            (episode/'meta.json').write_text(json.dumps({'termination':{'writer_errors':['disk full']}}))
            with patch.object(life,'verify',side_effect=AssertionError('must not decode')):
                result=life.audit_after_stop(episode,root,None,full_readback=False)
            self.assertEqual(result['exit_code'],4)
