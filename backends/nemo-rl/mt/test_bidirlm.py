"""Regression checks for BidirLM's packed hidden-state compatibility adapter."""

import unittest
from unittest.mock import MagicMock, Mock

from mt_reward.bidirlm import BidirLMHiddenStates


class BidirLMTests(unittest.TestCase):
    def make_model(self):
        model = Mock()
        model.layers = [Mock() for _ in range(26)]
        hidden = MagicMock()
        hidden.ndim, hidden.shape = 2, (5, 8)
        ids, mask = Mock(), Mock()
        ids.shape = (1, 4)
        mask.all.return_value = True
        return model, hidden, ids, mask

    def test_layer15_keeps_all_input_tokens_and_skips_internal_bos(self):
        model, hidden, ids, mask = self.make_model()
        def forward(**kwargs):
            # Upstream crashes if asked to return all intermediate states.
            self.assertFalse(kwargs["output_hidden_states"])
            hook = model.layers[15].register_forward_pre_hook.call_args.args[0]
            self.assertIsNone(hook(model.layers[15], (hidden,)))
        model.side_effect = forward
        out = BidirLMHiddenStates(model, 15)(input_ids=ids, attention_mask=mask)
        self.assertEqual(len(out["hidden_states"]), 27)
        self.assertIs(out["hidden_states"][15], hidden.__getitem__.return_value.unsqueeze.return_value)
        hidden.__getitem__.assert_called_once_with(slice(1, None))
        hidden.__getitem__.return_value.unsqueeze.assert_called_once_with(0)
        model.layers[15].register_forward_pre_hook.return_value.remove.assert_called_once()

    def test_final_layer_and_cleanup_after_failure(self):
        model, hidden, ids, mask = self.make_model()
        def forward(**kwargs):
            hook = model.norm.register_forward_hook.call_args.args[0]
            self.assertIsNone(hook(model.norm, (), hidden))
        model.side_effect = forward
        out = BidirLMHiddenStates(model, -1)(input_ids=ids, attention_mask=mask)
        self.assertIsNotNone(out["hidden_states"][-1])
        model.side_effect = RuntimeError("model failed")
        with self.assertRaisesRegex(RuntimeError, "model failed"):
            BidirLMHiddenStates(model, 15)(input_ids=ids, attention_mask=mask)
        model.layers[15].register_forward_pre_hook.return_value.remove.assert_called_once()

    def test_rejects_padding_and_unexpected_token_count(self):
        model, hidden, ids, mask = self.make_model()
        adapter = BidirLMHiddenStates(model, 15)
        mask.all.return_value = False
        with self.assertRaises(ValueError):
            adapter(input_ids=ids, attention_mask=mask)
        model.assert_not_called()
        mask.all.return_value = True
        hidden.shape = (4, 8)
        model.side_effect = lambda **kwargs: model.layers[15].register_forward_pre_hook.call_args.args[0](None, (hidden,))
        with self.assertRaisesRegex(RuntimeError, "token count"):
            adapter(input_ids=ids, attention_mask=mask)
