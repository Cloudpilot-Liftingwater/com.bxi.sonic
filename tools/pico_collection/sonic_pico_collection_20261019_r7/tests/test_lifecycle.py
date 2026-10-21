"""Offline fault injection: no camera, DDS, CAN, subprocess or robot startup."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock

import test_trial as t
import trial_lifecycle as life


class LifecycleTests(unittest.TestCase):
    def child(self,code=None):
        return dict(name='left_wrist_camera',process=mock.Mock(pid=321,poll=mock.Mock(return_value=code)),log='camera_left.log')

    def test_child_crash_names_camera_and_retains_return_code(self):
        failure=life.child_failure([self.child(-6)])
        self.assertEqual(failure['child'],'left_wrist_camera')
        self.assertEqual(failure['returncode'],-6)
        self.assertEqual(failure['log'],'camera_left.log')

    def test_monitor_handles_camera_crash_before_metadata(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            failure=life.wait_for_trial(root/'episode_0000',root,[self.child(134)])
            self.assertEqual(failure['reason'],'owned_process_exited')

    def test_finalization_race_still_verifies_saved_episode(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);episode=root/'episode_0000'
            failure=life.wait_for_trial(episode,root,[self.child(134)])
            # The worker flushes and renames during stop(), AFTER crash detection.
            episode.mkdir();(episode/'meta.json').write_text('{}')
            result=dict(readback='PASS',timing_review='PASS',episode_complete=True)
            with mock.patch.object(life,'verify',return_value=result) as verify,contextlib.redirect_stdout(io.StringIO()):
                report=life.audit_after_stop(episode,root,failure)
            verify.assert_called_once_with(episode,allow_incomplete=False)
            self.assertEqual(report['saved_data_checks'][0]['readback'],'PASS')
            self.assertEqual(report['exit_code'],4)  # Don't disguise camera failure as success.

    def test_complete_and_partial_and_corrupt_have_distinct_results(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);episode=root/'episode_0000'
            partial=root/'_interrupted'/'episode_0000_interrupted'
            partial.mkdir(parents=True);(partial/'meta.json').write_text('{}')
            with mock.patch.object(life,'verify',return_value=dict(readback='PASS',episode_complete=False,timing_review='INCOMPLETE')) as verify,contextlib.redirect_stdout(io.StringIO()):
                report=life.audit_after_stop(episode,root,{'reason':'disconnect'})
            verify.assert_called_once_with(partial,allow_incomplete=True)
            self.assertEqual(report['exit_code'],4)
            with mock.patch.object(life,'verify',side_effect=ValueError('bad JPEG')),contextlib.redirect_stdout(io.StringIO()):
                report=life.audit_after_stop(episode,root,{'reason':'disconnect'})
            self.assertEqual(report['saved_data_checks'][0]['readback'],'FAIL')
            self.assertTrue(partial.exists())

    def test_no_data_no_false_success_and_no_restart(self):
        with tempfile.TemporaryDirectory() as temp,contextlib.redirect_stdout(io.StringIO()):
            root=Path(temp)
            report=life.audit_after_stop(root/'episode_0000',root,{'reason':'startup_camera_failure'})
            self.assertTrue(report['no_saved_episode'])
            self.assertFalse(report['automatic_reconnect'])
            self.assertEqual(report['exit_code'],4)

    def test_normal_finished_and_timing_review_exit_codes(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);episode=root/'episode_0000';episode.mkdir();(episode/'meta.json').write_text('{}')
            self.assertIsNone(life.wait_for_trial(episode,root,[self.child()]))
            for timing,code in [('PASS',0),('REVIEW',2)]:
                with mock.patch.object(life,'verify',return_value=dict(readback='PASS',episode_complete=True,timing_review=timing)),contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(life.audit_after_stop(episode,root,None)['exit_code'],code)

    def test_timeout_is_bounded(self):
        with tempfile.TemporaryDirectory() as temp,mock.patch.object(life.time,'monotonic',side_effect=[0,181]):
            root=Path(temp)
            self.assertEqual(life.wait_for_trial(root/'episode_0000',root,[self.child()])['reason'],'trial_timeout')


class NodeShutdownTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.mod=t.load_node()

    def node(self,temp):
        node=t.NodeTests.node(self,temp);node.clock_offset=90.
        return node

    def test_single_saved_row_survives_shutdown_and_readback_is_incomplete(self):
        with tempfile.TemporaryDirectory() as temp:
            node=self.node(temp)
            node.recorder.start_episode(dict(task_id='TEST',instruction_text='SYNTHETIC ONLY'),runtime_metadata={})
            f=t.fixture();bundle,missing=t.select(f)
            self.assertFalse(missing)
            node._save_bundle(f[0],10.,bundle,t.JsonSample(9.9,99.9,{'current':{'name':'com.bxi.sonic/sonic_teleop'}}))
            node.close()
            path=next((node.source_root/'_interrupted').iterdir())
            with self.assertRaises(ValueError):t.verify(path)
            result=t.verify(path,allow_incomplete=True)
            self.assertEqual(result['decoded_images'],3)
            self.assertFalse(result['episode_complete'])
            self.assertEqual(result['timing_review'],'INCOMPLETE')
            self.assertIn('gripper_ingest',result['synchronization'])

    def test_silent_camera_aborts_without_stitch_or_restart(self):
        with tempfile.TemporaryDirectory() as temp:
            node=self.node(temp)
            try:
                node.recorder=mock.Mock(recording=True)
                f=t.fixture()
                for name,items in f[2].items():node.images[name].extend(items)
                with mock.patch.object(self.mod.time,'monotonic',return_value=12.5),mock.patch.object(node,'_abort') as abort:
                    node._capture_tick()
                    self.assertIn('camera_input_timeout:',abort.call_args.args[0])
            finally:node.recorder.recording=False;node.close()

    def test_ros_context_shutdown_after_executor_and_node_close(self):
        mod=self.mod;order=[];handlers={}
        node=mock.Mock();node.close.side_effect=lambda:order.append('close')
        node.destroy_node.side_effect=lambda:order.append('destroy')
        executor=mock.Mock();executor.shutdown.side_effect=lambda:order.append('executor_shutdown')
        def spin_once(**_):handlers[mod.signal.SIGINT](None,None)
        executor.spin_once.side_effect=spin_once
        with mock.patch.object(mod,'_arguments',return_value=types.SimpleNamespace(config='fake',data_id='fake')), \
             mock.patch.object(mod,'SourceCollectionNode',return_value=node), \
             mock.patch.object(mod,'MultiThreadedExecutor',return_value=executor), \
             mock.patch.object(mod.rclpy,'init') as init, \
             mock.patch.object(mod.rclpy,'ok',return_value=True), \
             mock.patch.object(mod.rclpy,'shutdown',side_effect=lambda:order.append('ros_shutdown')), \
             mock.patch.object(mod.signal,'getsignal',return_value='old_handler'), \
             mock.patch.object(mod.signal,'signal',side_effect=lambda s,h:handlers.update({s:h})):
            mod.main()
        self.assertEqual(order,['executor_shutdown','close','destroy','ros_shutdown'])
        self.assertEqual(init.call_args.kwargs['signal_handler_options'],mod.SignalHandlerOptions.NO)
        self.assertEqual(set(handlers.values()),{'old_handler'})


if __name__=='__main__':unittest.main()
