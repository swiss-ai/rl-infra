"""COMET reward routing and subprocess protocol without third-party weights."""

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from mt_reward.cometkiwi import CometKiwiReward
from mt_reward.kiwi_process import score_translation
from mt_reward.scoring import Example, TranslationReward
from mt_reward.rollouts import rollout_metrics


class CometKiwiTests(unittest.TestCase):
    def test_weighted_original_translation_without_reference_or_mask(self):
        components = Mock()
        components.language_matches.return_value = True
        components.mask.return_value = "MASKED"
        components.metricx.return_value = -5
        components.alignment.return_value = 0.5
        components.cometkiwi.return_value = 0.8
        example = Example("es", "de", "FUENTE", "Hallo Welt", "Hallo Welt")
        scorer = TranslationReward(components, cometkiwi_weight=20)
        row = scorer.score(example)
        self.assertAlmostEqual(row.reward, -5 + 10 + 20 + 16)
        self.assertEqual(row.cometkiwi, 0.8)
        components.cometkiwi.assert_called_once_with("FUENTE", "Hallo Welt")
        metrics = rollout_metrics({"message_log": [[{"role": "environment", "content":
                                   json.dumps({"mt_reward": row.asdict()})}]],
                                   "extra_env_info": [{"source_lang": "es", "target_lang": "de"}]})
        self.assertEqual(metrics["mt/es-to-de/cometkiwi/mean"], 0.8)
        components.cometkiwi.reset_mock()
        components.language_matches.return_value = False
        self.assertEqual(scorer.score(example).reward, -25)
        self.assertIsNone(scorer.score(example).cometkiwi)
        components.cometkiwi.assert_not_called()
        components.language_matches.return_value = True
        TranslationReward(components).score(example)
        components.cometkiwi.assert_not_called()
        for bad in (float("nan"), float("inf"), -0.1, 1.1):
            components.cometkiwi.return_value = bad
            with self.assertRaises(ValueError):
                scorer.score(example)

    def test_inference_preserves_comet_batch_structure_and_uses_predict_step(self):
        torch = Mock()
        tensor = Mock()
        torch.is_tensor.side_effect = lambda value: value is tensor
        for batch in ({"src_input_ids": tensor}, ({"input_ids": tensor, "length": 5},)):
            with self.subTest(batch_type=type(batch).__name__):
                model = Mock()
                model.prepare_sample.return_value = batch
                model.predict_step.return_value.scores.reshape.return_value = [0.85]
                value = score_translation(model, {"src": "Hola", "mt": "Hallo", "ref": "PRIVATE"},
                                          "cuda:0", torch)
                self.assertEqual(value, 0.85)
                model.prepare_sample.assert_called_once_with(
                    [{"src": "Hola", "mt": "Hallo"}], stage="predict")
                moved = model.predict_step.call_args.args[0]
                self.assertIs(type(moved), type(batch))
                inputs = moved[0] if isinstance(moved, tuple) else moved
                key = "input_ids" if isinstance(moved, tuple) else "src_input_ids"
                self.assertIs(inputs[key], tensor.to.return_value)
                model.predict_step.assert_called_once_with(moved, batch_idx=0)
                model.assert_not_called()
                model.forward.assert_not_called()
        tensor.to.assert_called_with("cuda:0")

    def test_process_protocol_and_shutdown(self):
        # Exercise real pipes and lifecycle with a lightweight fake interpreter.
        with tempfile.TemporaryDirectory() as directory:
            executable = Path(directory) / "kiwi-python"
            executable.write_text(
                '#!/usr/bin/env python3\nimport sys,json\n'
                'print(json.dumps({"ready": True}), flush=True)\n'
                'for line in sys.stdin:\n'
                ' row=json.loads(line)\n'
                ' assert set(row)=={"src","mt"}\n'
                ' print(json.dumps({"score": 0.75}), flush=True)\n')
            executable.chmod(0o755)
            scorer = CometKiwiReward({"cometkiwi_python": str(executable)})
            try:
                self.assertEqual(scorer.score("Hola", "Hallo"), 0.75)
                self.assertEqual(scorer.score("Otro", "Andere"), 0.75)
            finally:
                scorer.close()
            self.assertIsNotNone(scorer.process.poll())

    def test_errors_and_environment_isolation(self):
        process = Mock()
        process.stdout.readline.side_effect = ['{"ready": true}\n', '{"error": "inference failed"}\n', '']
        process.poll.return_value = 0
        with patch("mt_reward.cometkiwi.subprocess.Popen", return_value=process) as factory, \
                patch.dict(os.environ, {"PYTHONPATH": "/training", "CUDA_VISIBLE_DEVICES": "3"}):
            scorer = CometKiwiReward({"cometkiwi_python": "/kiwi/bin/python"})
            env = factory.call_args.kwargs["env"]
            self.assertNotIn("PYTHONPATH", env)
            self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "3")
            with self.assertRaisesRegex(RuntimeError, "inference failed"):
                scorer.score("src", "mt")
            with self.assertRaisesRegex(RuntimeError, "process exited"):
                scorer.score("src", "mt")
            scorer.close()
        with self.assertRaises(ValueError):
            CometKiwiReward({"cometkiwi_python": "python"})
