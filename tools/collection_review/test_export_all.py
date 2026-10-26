"""Export outcome isolation and no implicit promotion to training."""
import sys
from pathlib import Path
import unittest
from export_all import outcome_of


class OutcomeTests(unittest.TestCase):
    def test_saved(self):
        self.assertEqual(outcome_of({}, {'outcome':'saved'}, {}), 'saved')

    def test_discard_flag_wins_over_conflicting_saved_result(self):
        self.assertEqual(outcome_of({'discarded_episode_indices':[0]}, {'outcome':'saved'}, {}), 'discarded')

    def test_interrupted_and_missing_reports_are_not_saved(self):
        self.assertEqual(outcome_of({}, {'outcome':'interrupted'}, {}), 'interrupted')
        self.assertEqual(outcome_of({}, {}, {}), 'unknown')

    def test_discard_inspection_cannot_be_training_export(self):
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'gr00t_dataset'))
        from export_dataset import export
        with self.assertRaisesRegex(ValueError, 'inspection_only'):
            export([], '/unused', include_discarded=True)


if __name__=='__main__':unittest.main()
