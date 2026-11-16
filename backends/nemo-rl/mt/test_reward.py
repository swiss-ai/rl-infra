"""CPU tests: python -m unittest discover -s backends/nemo-rl/mt -v."""

from dataclasses import replace
import json
import math
from pathlib import Path
import tempfile
import sys
import unittest
from unittest.mock import MagicMock, Mock, patch

from sacrebleu.metrics import CHRF

from mt_reward.scoring import Example, TranslationReward, alignment_f1
from mt_reward.models import WalarComponents, ReferenceAlignment
from mt_reward.metricx import MetricXReward
from mt_reward.rollouts import examples_from_rollouts, rollout_metrics
from mt_reward.runtime import load_scorer_config, worker_options
from mt_reward.worker_core import RewardWorker


class RewardTests(unittest.TestCase):
    def setUp(self):
        self.example = Example("en", "de", "Hello world.", "Hallo Welt.", "Hallo Welt.")
        self.components = Mock()
        self.components.language_matches.return_value = True
        self.components.mask.return_value = "Hallo"
        self.components.metricx.return_value = -5
        self.components.alignment.return_value = 0.5
        self.reward = TranslationReward(self.components)

    def test_perfect_reference_and_weighting(self):
        score = self.reward.score(self.example)
        self.assertAlmostEqual(score.chrf_pp, 100)
        self.assertAlmostEqual(score.reward, -5 + 20 * 0.5 + 20)
        self.components.metricx.assert_called_once_with("Hello world.", "Hallo Welt.", "Hallo Welt.")
        self.components.alignment.assert_called_once_with("Hallo Welt.", "Hallo", "de", "de")

    def test_chrf_is_sentence_level_plus_plus_on_unmasked_output(self):
        example = replace(self.example, hypothesis="Hallo schöne Welt.")
        expected = CHRF(char_order=6, word_order=2, beta=2).sentence_score(
            example.hypothesis, [example.reference]).score
        score = self.reward.score(example)
        self.assertAlmostEqual(score.chrf_pp, expected)
        self.assertAlmostEqual(score.reward, 5 + 20 * expected / 100)

    def test_wrong_language_overrides_perfect_reference(self):
        self.components.language_matches.return_value = False
        score = self.reward.score(self.example)
        self.assertEqual(score.reward, -25)
        self.assertFalse(score.language_alignment)
        self.assertIsNone(score.metricx)
        self.components.metricx.assert_not_called()
        self.components.mask.assert_not_called()

    def test_empty_output_is_penalized(self):
        score = self.reward.score(replace(self.example, hypothesis=""))
        self.assertEqual(score.reward, -25)
        self.assertEqual(score.chrf_pp, 0)
        self.components.language_matches.assert_not_called()

    def test_empty_mask_has_zero_alignment(self):
        self.components.mask.return_value = ""
        score = self.reward.score(self.example)
        self.assertEqual(score.word_alignment, 0)
        self.components.alignment.assert_not_called()

    def test_zero_chrf_weight_keeps_metricx_and_alignment(self):
        score = TranslationReward(self.components, chrf_weight=0).score(self.example)
        self.assertEqual(score.reward, 5)

    def test_alignment_uses_unique_word_coverage(self):
        pairs = [(0, 0), (0, 1), (1, 0), (1, 1)]
        self.assertEqual(alignment_f1(pairs, [0, 0], [0, 0], 2, 2), 0.5)
        self.assertEqual(alignment_f1([], [0], [0], 1, 1), 0)
        self.assertEqual(alignment_f1([], [], [0], 0, 1), 0)

    def test_ctfalign_loader_uses_configured_encoder_and_device(self):
        ctfalign, fasttext, hub = Mock(), Mock(), Mock()
        hub.snapshot_download.return_value = "/cache/bidirlm-snapshot"
        fasttext.load_model.return_value.get_labels.return_value = ["__label__deu_Latn"]
        upstream = Mock()
        upstream.MaskLID.return_value.model.get_labels.return_value = ["__label__deu_Latn"]
        config = {"walar_dir": "/walar", "device": "cuda:0",
                  "languages": {"de": "deu_Latn"}, "glotlid_model": "/lid",
                  "masklid_model": "/mask"}
        with patch.dict(sys.modules, {"ctfalign": ctfalign, "fasttext": fasttext, "huggingface_hub": hub}), \
                patch("mt_reward.models.load_module", return_value=upstream), \
                patch("mt_reward.models.MetricXReward"):
            WalarComponents(config)
            ctfalign.WordAligner.from_huggingface.assert_called_with(
                "/cache/bidirlm-snapshot", method="ctfalign", mode="argmax",
                units="words", device="cuda:0", layer=15,
                granularity="documents", trust_remote_code=True)
            hub.snapshot_download.assert_called_once_with(
                "BidirLM/BidirLM-1B-Embedding", revision="transformers-v4")
            hub.snapshot_download.reset_mock()
            WalarComponents({**config, "alignment_model": "/local/encoder", "alignment_layer": 12})
            self.assertEqual(ctfalign.WordAligner.from_huggingface.call_args.args, ("/local/encoder",))
            self.assertEqual(ctfalign.WordAligner.from_huggingface.call_args.kwargs["layer"], 12)
            hub.snapshot_download.assert_not_called()
            WalarComponents({**config, "alignment_model": "/local/bidirlm",
                             "alignment_layer": 15, "alignment_trust_remote_code": True})
            self.assertTrue(ctfalign.WordAligner.from_huggingface.call_args.kwargs["trust_remote_code"])
            hub.snapshot_download.assert_not_called()

    def test_bidirlm_preserves_gemma_tokenizer_and_wraps_hidden_states(self):
        ctfalign, transformers, hub = Mock(), Mock(), Mock()
        hub.snapshot_download.return_value = "/cache/bidirlm"
        encoder = ctfalign.WordAligner.from_huggingface.return_value.encoder
        encoder.model.config.model_type = "bidirlm"
        encoder.model.layers = [Mock() for _ in range(26)]
        encoder.layer = 15
        original_model = encoder.model
        with patch.dict(sys.modules, {"ctfalign": ctfalign, "transformers": transformers,
                                      "huggingface_hub": hub}):
            ReferenceAlignment({"languages": {"de": "deu_Latn"}})
        transformers.AutoTokenizer.from_pretrained.assert_called_once_with(
            "/cache/bidirlm", fix_mistral_regex=False)
        self.assertIs(encoder.model.model, original_model)
        self.assertEqual(encoder.model.layer, 15)

    def test_ctfalign_uses_word_pairs_and_unique_coverage(self):
        models = object.__new__(ReferenceAlignment)
        models.languages = {"de": "deu_Latn"}
        models.aligner = Mock()
        models.aligner.align.return_value = [(0, 0), (0, 1), (0, 1)]
        self.assertAlmostEqual(models.alignment("Hallo Welt", "Hallo du Welt", "de", "de"), 4 / 7)
        models.aligner.align.assert_called_once_with("Hallo Welt", "Hallo du Welt")
        models.aligner.align.reset_mock()
        self.assertEqual(models.alignment("", "Hallo", "de", "de"), 0)
        self.assertEqual(models.alignment("Hallo", "", "de", "de"), 0)
        models.aligner.align.assert_not_called()
        models.aligner.align.return_value = []
        self.assertEqual(models.alignment("Hallo", "Welt", "de", "de"), 0)

    def test_ctfalign_preserves_chinese_segmentation(self):
        models = object.__new__(ReferenceAlignment)
        models.languages = {"zh": "cmn_Hans"}
        models.han = Mock(side_effect=[["你好", "，", "世界"], ["世界", "。"]])
        models.aligner = Mock()
        models.aligner.align.return_value = [(1, 0)]
        self.assertAlmostEqual(models.alignment("你好，世界", "世界。", "zh", "zh"), 2 / 3)
        models.aligner.align.assert_called_once_with("你好 世界", "世界")

    def test_reference_alignment_does_not_depend_on_source(self):
        self.reward.score(replace(self.example, source="A differently phrased source", source_lang="fr"))
        self.components.alignment.assert_called_once_with("Hallo Welt.", "Hallo", "de", "de")

    def test_metricx_negates_error_and_includes_reference(self):
        scorer = object.__new__(MetricXReward)
        scorer.device, scorer.max_length = "cpu", 1536
        scorer.torch = MagicMock()
        ids, mask = MagicMock(), MagicMock()
        scorer.tokenizer = Mock(return_value={"input_ids": ids, "attention_mask": mask})
        scorer.model = Mock()
        scorer.model.return_value.predictions.reshape.return_value = [7.0]
        self.assertEqual(scorer.score("source", "candidate", "reference"), -7)
        scorer.model.assert_called_once_with(input_ids=ids.__getitem__.return_value.to.return_value,
                                             attention_mask=mask.__getitem__.return_value.to.return_value,
                                             use_cache=False)
        scorer.tokenizer.assert_called_once_with(
            "source: source candidate: candidate reference: reference",
            max_length=1536, truncation=True, padding=False, return_tensors="pt")
        ids.__getitem__.assert_called_once_with((slice(None), slice(None, -1)))
        mask.__getitem__.assert_called_once_with((slice(None), slice(None, -1)))
        for reference in (None, "", " "):
            with self.subTest(reference=reference), self.assertRaises(ValueError):
                scorer.score("source", "candidate", reference)
        for error in (float("nan"), -1, 26):
            scorer.model.return_value.predictions.reshape.return_value = [error]
            with self.subTest(error=error), self.assertRaises(ValueError):
                scorer.score("source", "candidate", "reference")

    def test_reference_reaches_metricx_through_component_adapter(self):
        models = object.__new__(WalarComponents)
        models.metricx_scorer = Mock()
        models.metricx_scorer.score.return_value = -3
        self.assertEqual(models.metricx("source", "candidate", "reference"), -3)
        models.metricx_scorer.score.assert_called_once_with("source", "candidate", "reference")

    def test_metricx_loader_rejects_incomplete_weights_and_invalid_limit(self):
        upstream, torch, transformers = Mock(), Mock(), Mock()
        model = Mock()
        info = {"missing_keys": [], "mismatched_keys": [], "error_msgs": []}
        config = {"walar_dir": "/walar", "device": "cpu"}
        with patch.dict(sys.modules, {"torch": torch, "transformers": transformers}), \
                patch("mt_reward.models.load_module", return_value=upstream):
            upstream.MT5ForRegression.from_pretrained.return_value = (model, info)
            scorer = MetricXReward(config)
            self.assertEqual(scorer.max_length, 1536)
            model.to.assert_called_with("cpu")
            for key in info:
                upstream.MT5ForRegression.from_pretrained.return_value = (
                    model, {**info, key: ["bad weight"]})
                with self.subTest(key=key), self.assertRaises(RuntimeError):
                    MetricXReward(config)
            with self.assertRaises(ValueError):
                MetricXReward({**config, "max_length": 2048})

    def test_language_detection_batch_api_and_label_mapping(self):
        models = object.__new__(WalarComponents)
        models.languages = {"de": "deu_Latn", "en": "eng_Latn"}
        models.lid = Mock()
        models.lid.predict.return_value = ([["__label__deu_Latn"]], [[0.99]])
        self.assertTrue(models.language_matches("Hallo\nWelt.", "de"))
        models.lid.predict.assert_called_with(["Hallo Welt."])
        self.assertFalse(models.language_matches("Hallo Welt.", "en"))

    def test_actor_configuration_from_file_or_inline(self):
        scorer = {"chrf_weight": 20}
        self.assertEqual(load_scorer_config({"scorer": scorer}), scorer)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scorer.json"
            path.write_text(json.dumps(scorer))
            self.assertEqual(load_scorer_config({"scorer_config": str(path)}), scorer)
        with self.assertRaises(ValueError):
            load_scorer_config({"scorer": scorer, "scorer_config": "unused"})
        with self.assertRaises(ValueError):
            load_scorer_config({"scorer": []})

    def test_metricx_floor_still_uses_declared_additive_formula(self):
        self.components.metricx.return_value = -25
        self.assertEqual(self.reward.score(self.example).reward, 5)

    def test_invalid_reference_and_nonfinite_scores_fail(self):
        for reference in ("", " ", None):
            with self.subTest(reference=reference), self.assertRaises(ValueError):
                replace(self.example, reference=reference)
        for value in (math.nan, math.inf, 1, -26):
            self.components.metricx.return_value = value
            with self.subTest(metricx=value), self.assertRaises(ValueError):
                self.reward.score(self.example)
        for weight in (-1, math.inf, math.nan):
            with self.subTest(weight=weight), self.assertRaises(ValueError):
                TranslationReward(self.components, chrf_weight=weight)

    def test_worker_batch_order_and_error_propagation(self):
        with patch("mt_reward.models.WalarComponents", return_value=self.components) as factory:
            worker = RewardWorker({"device": "cuda:3", "chrf_weight": 10})
        self.assertEqual(factory.call_args.args[0]["device"], "cuda:0")
        self.assertTrue(worker.ready())
        examples = [self.example, replace(self.example, hypothesis="")]
        scores = worker.score([e.__dict__ for e in examples])
        self.assertEqual([score["reward"] for score in scores], [15, -25])
        self.assertEqual(worker.score([]), [])
        self.components.metricx.side_effect = RuntimeError("model failed")
        with self.assertRaisesRegex(RuntimeError, "model failed"):
            worker.score([self.example.__dict__])

    def test_worker_resource_and_runtime_isolation(self):
        config = {"scorer_python": "/scratch/mt-venv/bin/python",
                  "resources": {"num_nodes": 1, "gpus_per_node": 1}}
        variables = {"CUDA_VISIBLE_DEVICES": "0,1,2,3", "VIRTUAL_ENV": "/training",
                     "PYTHONPATH": "/mt", "UV_PROJECT_ENVIRONMENT": "/training",
                     "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1"}
        options = worker_options(config, variables)
        self.assertEqual(options["num_gpus"], 1)
        self.assertEqual(options["num_cpus"], 4)
        runtime = options["runtime_env"]
        self.assertEqual(runtime["py_executable"], config["scorer_python"])
        self.assertEqual(runtime["env_vars"]["VIRTUAL_ENV"], "/scratch/mt-venv")
        self.assertEqual(runtime["env_vars"]["PYTHONPATH"], "/mt")
        self.assertNotIn("CUDA_VISIBLE_DEVICES", runtime["env_vars"])
        self.assertNotIn("UV_PROJECT_ENVIRONMENT", runtime["env_vars"])
        self.assertNotIn("RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES", runtime["env_vars"])
        self.assertEqual(variables["VIRTUAL_ENV"], "/training")
        with self.assertRaises(ValueError):
            worker_options({**config, "scorer_python": "python"}, {})
        with self.assertRaises(ValueError):
            worker_options({**config, "resources": {"num_nodes": 1, "gpus_per_node": 4}}, {})

    def test_rollout_metadata_and_direction_metrics(self):
        info = {key: value for key, value in self.example.__dict__.items() if key != "hypothesis"}
        messages = [{"role": "user", "content": "Translate"},
                    {"role": "assistant", "content": self.example.hypothesis}]
        self.assertEqual(examples_from_rollouts([messages], [info]), [self.example])
        good = self.reward.score(self.example).asdict()
        bad = self.reward.score(replace(self.example, hypothesis="")).asdict()
        batch = {"message_log": [messages + [{"role": "environment", "content":
                 json.dumps({"mt_reward": score})}] for score in (good, bad)],
                 "extra_env_info": [info, info]}
        metrics = rollout_metrics(batch)
        self.assertEqual(metrics["mt/reward/mean"], 0)
        self.assertEqual(metrics["mt/en-to-de/language_alignment/mean"], 0.5)
        self.assertEqual(metrics["mt/metricx/mean"], -5)
        self.assertEqual(metrics["mt/num_samples"], 2)
        with self.assertRaises(ValueError):
            examples_from_rollouts([messages], [])
        with self.assertRaises(ValueError):
            examples_from_rollouts([messages[:1]], [info])



if __name__ == "__main__":
    unittest.main()
