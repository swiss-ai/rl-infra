"""Translation prompt and processor tests without model downloads."""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from mt_reward.prompts import format_example
from mt_reward.processor import translation_data_processor


class DataTests(unittest.TestCase):
    def setUp(self):
        self.row = {"src": "Buenos días.", "tgt": "REFERENZ_NUR_FUER_REWARD",
                    "source_language": "es", "target_language": "de",
                    "labse_similarity": 0.95}

    def test_names_variety_and_no_reference_leakage(self):
        examples = [format_example(self.row, index) for index in range(64)]
        self.assertEqual(len({example["template_id"] for example in examples}), 4)
        for example in examples:
            self.assertIn("Spanish", example["prompt"])
            self.assertIn("German", example["prompt"])
            self.assertNotIn(self.row["tgt"], example["prompt"])
        self.assertEqual(examples[5], format_example(self.row, 5))
        self.assertNotEqual([x["template_id"] for x in examples],
                            [format_example(self.row, i, seed=7)["template_id"] for i in range(64)])

    def test_unknown_language_and_missing_reference_fail(self):
        for bad in ({**self.row, "source_language": "xx"}, {**self.row, "tgt": ""}):
            with self.assertRaises(ValueError):
                format_example(bad, 0)

    def test_internal_schema_matches_input_aliases(self):
        internal = {"source": self.row["src"], "reference": self.row["tgt"],
                    "source_lang": "es", "target_lang": "de"}
        self.assertEqual(format_example(internal, 0), format_example(self.row, 0))

    def test_processor_keeps_reference_in_metadata_and_disables_thinking(self):
        datum = {**format_example(self.row, 0), "task_name": "mt"}
        tokenizer = Mock()
        tokenizer.apply_chat_template.return_value = "FORMATTED_PROMPT"
        tokenizer.return_value = {"input_ids": [[1, 2, 3]]}
        spec = SimpleNamespace(system_prompt=None)
        result = translation_data_processor(datum, spec, tokenizer, 4096, 0)
        self.assertEqual(result["extra_env_info"]["reference"], self.row["tgt"])
        self.assertEqual(result["extra_env_info"]["template_id"], datum["template_id"])
        self.assertEqual(result["length"], 3)
        call = tokenizer.apply_chat_template.call_args
        self.assertFalse(call.kwargs["enable_thinking"])
        self.assertNotIn(self.row["tgt"], str(call.args[0]))
        with self.assertRaises(ValueError):
            translation_data_processor(datum, spec, tokenizer, 3, 0)


if __name__ == "__main__":
    unittest.main()
