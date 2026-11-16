"""Checks for the full-reward diagnostic's reporting and ranking assertions."""

import unittest
from unittest.mock import Mock

from check_reward import builtin_checks, reward_examples, score_example
from mt_reward.scoring import Example, TranslationReward


class RewardCheckTests(unittest.TestCase):
    def test_reports_actual_weighted_terms_and_gate_override(self):
        components = Mock()
        components.language_matches.return_value = True
        components.metricx.return_value = -3
        components.alignment.return_value = 0.7
        scorer = TranslationReward(components, alignment_weight=8, chrf_weight=5)
        worker = Mock()
        worker.score.side_effect = lambda rows: [scorer.score(Example(**row)).asdict() for row in rows]
        example = reward_examples()[0]
        row = score_example(worker, example, {"alignment_weight": 8, "chrf_weight": 5})
        self.assertAlmostEqual(row["reward"], row["metricx"] + row["alignment_bonus"] + row["chrf_bonus"])
        self.assertAlmostEqual(row["alignment_bonus"], 5.6)
        self.assertEqual(row["chrf_bonus"], 5)
        self.assertNotIn("name", worker.score.call_args.args[0][0])
        components.language_matches.return_value = False
        row = score_example(worker, example, {})
        self.assertEqual(row["reward"], -25)
        self.assertEqual(row["alignment_bonus"], 0)
        self.assertEqual(row["chrf_bonus"], 0)

    def test_checks_distinguish_metricx_and_total_reward_rankings(self):
        rows = []
        for example in reward_examples():
            gated = example["name"] in ("wrong_language", "empty_output")
            bad = example["name"] in ("unrelated_document", "incorrect_facts")
            rows.append({**example, "language_alignment": not gated,
                         "metricx": None if gated else (-10 if bad else -1),
                         "word_alignment": None if gated else 1,
                         "chrf_pp": 100, "reward": -25 if gated else (5 if bad else 20)})
        self.assertTrue(all(builtin_checks(rows).values()))
        next(row for row in rows if row["name"] == "incorrect_facts")["reward"] = 21
        checks = builtin_checks(rows)
        self.assertTrue(checks["MetricX prefers good paraphrase to incorrect_facts"])
        self.assertFalse(checks["Total reward prefers good paraphrase to incorrect_facts"])
