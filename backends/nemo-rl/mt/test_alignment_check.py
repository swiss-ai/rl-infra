"""Verify the document check reports the production coverage and reward bonus."""

import unittest
from unittest.mock import Mock

from check_alignment import document_examples, evaluate
from mt_reward.models import ReferenceAlignment, WalarComponents


class AlignmentCheckTests(unittest.TestCase):
    def test_report_uses_production_coverage_and_configured_weight(self):
        scorer = object.__new__(ReferenceAlignment)
        scorer.languages = {"de": "deu_Latn"}
        scorer.aligner = Mock()
        scorer.aligner.align.return_value = [(0, 0), (0, 1), (0, 1)]
        row = evaluate(scorer, {"reference": "Hallo Welt", "hypothesis": "Hallo du Welt",
                                "target_lang": "de"}, 10)
        self.assertEqual(row["reference_coverage"], 0.5)
        self.assertEqual(row["hypothesis_coverage"], 2 / 3)
        self.assertAlmostEqual(row["word_alignment"], 4 / 7)
        self.assertAlmostEqual(row["alignment_bonus"], 40 / 7)
        self.assertEqual(row["aligned_words"][0], ["Hallo", "Hallo"])
        production = object.__new__(WalarComponents)
        production.alignment_scorer = scorer
        self.assertEqual(production.alignment("Hallo Welt", "Hallo du Welt", "de", "de"),
                         row["word_alignment"])

    def test_default_documents_and_invalid_input(self):
        examples = document_examples()
        self.assertEqual(len(examples), 4)
        self.assertTrue(all(len(row["hypothesis"].split()) >= 60 for row in examples))
        self.assertTrue(all(row["reference"] == examples[0]["reference"] for row in examples))
        with self.assertRaises(ValueError):
            evaluate(Mock(), {"target_lang": "de", "reference": "Hallo", "hypothesis": ""}, 20)
