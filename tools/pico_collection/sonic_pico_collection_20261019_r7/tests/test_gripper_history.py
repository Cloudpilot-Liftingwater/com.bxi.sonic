"""Regression for delayed/ineligible packets while RGB anchors wait 250 ms."""
import json
import tempfile
import types
import unittest
from unittest import mock

import test_trial as t
from real_robot_collection.gripper_history import GripperHistory, blocked_by_intervals
from real_robot_collection.wrist_diagnostic import convert_raw
from real_robot_collection.source_sync import nearest
from real_robot_collection.synchronization import JsonSample


def raw_at(stamp=100.):
    return t.fixture(stamp=stamp)[1]['gripper'][0].raw


def ingest(history, value, received=10.08):
    return history.ingest(json.dumps(value), received, 90.)


class SourceHistoryTests(unittest.TestCase):
    def test_late_transport_is_not_stale_source(self):
        raw=raw_at(); history=GripperHistory()
        samples,error=ingest(history,raw)
        self.assertIsNone(error)
        self.assertEqual(samples[0].raw,raw)
        self.assertEqual(samples[0].received_monotonic,10.08)
        self.assertAlmostEqual(history.summary()['publication_to_callback_ms']['median'],85.)
        self.assertEqual(history.counts['transport_over_50ms'],1)
        with self.assertRaisesRegex(ValueError,'stale publication'):convert_raw(raw,10.08)

    def test_source_feedback_and_command_staleness_still_rejected(self):
        for side in (0,1):
            for key in ('feedback','command'):
                raw=raw_at()
                if key=='feedback':raw['samples'][side]['feedback_monotonic']=9.93
                else:raw['samples'][side]['command']['sent_monotonic']=9.93
                history=GripperHistory();samples,error=ingest(history,raw)
                self.assertIsNone(samples)
                self.assertIn('at source',error)
                self.assertTrue(history.invalid_intervals[-1]['source_stamp_known'])

    def test_future_or_nan_source_feedback_is_rejected(self):
        for stamp in (10.1,float('nan'),float('inf')):
            raw=raw_at();raw['samples'][0]['feedback_monotonic']=stamp
            self.assertIsNone(ingest(GripperHistory(),raw)[0])

    def test_transport_is_bounded_not_unlimited(self):
        history=GripperHistory()
        samples,error=ingest(history,raw_at(),10.8)
        self.assertIsNone(samples)
        self.assertEqual(error,'transport exceeds buffered window')

    def test_invalid_publication_clock_is_not_trusted(self):
        for kind in ('future','nan','offset','stamp_type','nanosec'):
            raw=raw_at()
            if kind=='future':raw['monotonic']=10.1
            elif kind=='nan':raw['monotonic']=float('nan')
            elif kind=='offset':raw['stamp']['sec']+=2
            elif kind=='stamp_type':raw['stamp']['sec']='99'
            else:raw['stamp']['nanosec']=1000000000
            history=GripperHistory(); samples,error=ingest(history,raw)
            self.assertIsNone(samples,kind)
            self.assertFalse(history.invalid_intervals[-1]['source_stamp_known'])

    def test_raw_identity_range_and_eligibility_guards_remain(self):
        for kind in ('bus','profile','fraction','range','pause','reasons','missing_command','bad_item'):
            raw=raw_at()
            if kind=='bus':raw['samples'][1]['bus']=5
            elif kind=='profile':raw['samples'][1]['profile']='bxi_colleague_rad_v1'
            elif kind=='fraction':raw['samples'][1]['closure_fraction']=.9
            elif kind=='range':
                raw['samples'][1].update(position_rad=10.,closure_fraction=-1.)
            elif kind=='pause':raw['training_eligible']=False
            elif kind=='reasons':raw['ineligible_reasons']=['right_control_inactive']
            elif kind=='missing_command':raw['samples'][1]['command']=None
            else:raw['samples'][1]=None
            self.assertIsNone(ingest(GripperHistory(),raw)[0],kind)

    def test_malformed_json_never_throws_into_executor(self):
        for text in ('{','null','[]','42','{"monotonic":0,"stamp":null}'):
            history=GripperHistory()
            result,reason=history.ingest(text,10.08,90.)
            self.assertIsNone(result)
            self.assertTrue(reason)
            self.assertFalse(history.invalid_intervals[-1]['source_stamp_known'])

    def test_pause_barrier_preserves_earlier_rows_but_blocks_fallback(self):
        history=GripperHistory();raw=raw_at(100.02)
        raw.update(training_eligible=False,ineligible_reasons=['pose_paused_or_stale'])
        ingest(history,raw)
        barriers=history.invalid_intervals
        self.assertFalse(blocked_by_intervals(99.995,100.,barriers))
        self.assertTrue(blocked_by_intervals(99.995,100.02,barriers))
        # A queued older good packet, or equal timestamp, cannot clear a pause.
        self.assertTrue(blocked_by_intervals(100.005,100.02,barriers))
        self.assertTrue(blocked_by_intervals(barriers[0]['start_unix'],100.02,barriers))
        self.assertFalse(blocked_by_intervals(100.025,100.03,barriers))
        self.assertEqual(history.summary()['ineligible_reasons'],{'pose_paused_or_stale':1})

    def test_unknown_interval_requires_new_source_after_receipt(self):
        history=GripperHistory();history.ingest('{',10.08,90.)
        barriers=history.invalid_intervals
        self.assertTrue(blocked_by_intervals(99.995,100.,barriers))
        self.assertTrue(blocked_by_intervals(100.07,100.08,barriers))
        self.assertFalse(blocked_by_intervals(100.09,100.1,barriers))

    def test_reset_counters_does_not_clear_barriers(self):
        history=GripperHistory();history.ingest('{',10.08,90.)
        history.reset_diagnostics()
        self.assertFalse(history.counts)
        self.assertEqual(len(history.invalid_intervals),1)
        self.assertNotIn('invalid_intervals',history.summary(include_intervals=False))

    def test_matcher_rejects_invalid_interval_and_accepts_recovery(self):
        f=t.fixture(stamp=100.02);anchor,hist,images,pico,smpl=f
        hist['gripper']=[t.fixture()[1]['gripper'][0]]
        history=GripperHistory();raw=raw_at(100.015);raw['training_eligible']=False
        ingest(history,raw)
        hist['gripper_invalid_intervals']=list(history.invalid_intervals)
        self.assertIn('gripper_invalid_source_interval',t.select(f)[1])
        hist['gripper']=[t.fixture(stamp=100.02)[1]['gripper'][0]]
        self.assertNotIn('gripper_invalid_source_interval',t.select(f)[1])

    def test_source_silence_and_future_data_cannot_fill_gap(self):
        old=t.fixture()[1]['gripper'][0]
        self.assertIsNone(nearest([old],100.04,.03,causal=True))
        self.assertIsNone(nearest([old],99.98,.03,causal=True))

    def test_reordered_50hz_stream_with_80ms_delays_matches_all_30hz_anchors(self):
        # No ROS and no fabricated saved rows: replay only the ingest/selection.
        history=GripperHistory();pending=[];valid=[]
        for i in range(65):
            raw=raw_at(100.+i*.02)
            delay=.08 if i%5==0 else .005
            pending.append((raw['monotonic']+delay,raw))
        pending.sort(key=lambda p:p[0])
        for i in range(1,31):
            anchor=100.+i/30
            while pending and pending[0][0] <= anchor-90.+.25:
                receipt,raw=pending.pop(0)
                samples,error=ingest(history,raw,receipt)
                self.assertIsNone(error)
                valid.append(samples[0])
            self.assertIsNotNone(nearest(valid,anchor,.03,causal=True))
        self.assertGreater(history.counts['transport_over_50ms'],0)


class CallbackRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.mod=t.load_node()

    def node(self,temp):
        n=t.NodeTests.node(self,temp);n.clock_offset=90.
        return n

    def feed(self,node,raw,receipt):
        with mock.patch.object(self.mod.time,'monotonic',return_value=receipt):
            node._on_native_gripper(types.SimpleNamespace(data=json.dumps(raw)))

    def test_late_packet_does_not_clear_valid_history(self):
        with tempfile.TemporaryDirectory() as temp:
            n=self.node(temp)
            try:
                self.feed(n,raw_at(),10.)
                self.feed(n,raw_at(100.02),10.08)
                self.assertEqual(len(n.gripper_states),2)
                self.assertEqual(len(n.gripper_commands),2)
                self.assertEqual(n.gripper_states[0].stamp_unix,99.995)
            finally:n.close()

    def test_later_source_fault_does_not_erase_pending_older_camera_row(self):
        with tempfile.TemporaryDirectory() as temp:
            n=self.node(temp)
            try:
                f=t.fixture(1,100.05);anchor,hist,images,pico,smpl=f
                for attr,key in (('states','state'),('commands','command'),('imus','imu'),('sonic_inferences','inference')):
                    getattr(n,attr).extend(hist[key])
                for name,items in images.items():n.images[name].extend(items)
                self.feed(n,hist['gripper'][0].raw,10.13)  # 85 ms delayed, source valid
                bad=raw_at(100.15);bad['samples'][1]['feedback_monotonic']=9.9
                self.feed(n,bad,10.16)  # invalid AFTER the pending head anchor
                self.assertEqual(len(n.gripper_states),1)
                n.pico.after.return_value=[pico];n.smpl.after.return_value=[smpl]
                n.status_history.append(JsonSample(9.95,99.95,{'current':{'name':'com.bxi.sonic/sonic_teleop'}}))
                n.recorder.start_episode(dict(task_id='TEST',instruction_text='SYNTHETIC ONLY'),runtime_metadata={})
                n.source_buffer.reset(100.)
                with mock.patch.object(self.mod.time,'time',return_value=100.31),mock.patch.object(self.mod.time,'monotonic',return_value=10.31):
                    n._capture_tick()
                n.recorder.queue.join()
                self.assertEqual(n.recorder.frame_count,1)
                self.assertEqual(n.last_missing,[])
            finally:n.close()

    def test_ready_blocks_pause_immediately_without_erasing_history(self):
        with tempfile.TemporaryDirectory() as temp:
            n=self.node(temp)
            try:
                self.feed(n,raw_at(),10.)
                paused=raw_at(100.01);paused['training_eligible']=False
                self.feed(n,paused,10.01)
                with mock.patch.object(self.mod.RealCollectionNode,'_ready',return_value=[]),mock.patch.object(n,'_clock_ok',return_value=True):
                    self.assertIn('valid_gripper_source_history',n._ready(10.01))
                self.assertEqual(len(n.gripper_states),1)
            finally:n.close()


if __name__=='__main__':unittest.main()
