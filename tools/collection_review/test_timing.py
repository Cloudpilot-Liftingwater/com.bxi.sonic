import json,tempfile,unittest
from pathlib import Path
from timing import analyze_episode
class TimingTests(unittest.TestCase):
 def run_rows(self,rows):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);(p/'source.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows));return analyze_episode(p)
 def rows(self,offset):return [dict(inference_sequence=i,inference_stamp_unix=100+i*.02,received_monotonic=10+i*.02,source_stamps={'state':100+i*.02+offset}) for i in range(100)]
 def test_old_second_scale_mismatch_without_v15_fields(self):
  r=self.run_rows(self.rows(5.1));self.assertEqual(r['status'],'SECOND_SCALE_DELAY_OBSERVED');self.assertIsNone(r['writer_queue_age_ms'])
  self.assertAlmostEqual(r['sensor_minus_token_ms']['state']['median'],5100)
 def test_current_records_show_queue_and_receipt(self):
  rows=self.rows(.001)
  for row in rows:row.update(writer_queue_age_s=.004,inference_receipt_delay_s=.002)
  r=self.run_rows(rows);self.assertEqual(r['status'],'NO_SECOND_SCALE_DELAY_OBSERVED');self.assertEqual(r['writer_queue_age_ms']['max'],4)
  self.assertAlmostEqual(r['writer_span_hz_including_pauses'],50)
 def test_pause_is_not_hidden_by_median_interval(self):
  rows=self.rows(0)
  for r in rows[50:]:r['inference_stamp_unix']+=5;r['received_monotonic']+=5;r['source_stamps']['state']+=5
  r=self.run_rows(rows);self.assertAlmostEqual(r['source_span_hz_including_pauses'],99/6.98);self.assertEqual(r['status'],'NO_SECOND_SCALE_DELAY_OBSERVED')
 def test_sequence_reset_is_not_negative_loss(self):
  rows=self.rows(0);rows[50]['inference_sequence']=-1;r=self.run_rows(rows);self.assertEqual(r['sequence_resets'],1)
 def test_empty_and_missing(self):
  self.assertEqual(self.run_rows([])['status'],'EMPTY')
  with tempfile.TemporaryDirectory() as d:self.assertEqual(analyze_episode(d)['status'],'NO_SOURCE_TIMESTAMPS')
if __name__=='__main__':unittest.main()
