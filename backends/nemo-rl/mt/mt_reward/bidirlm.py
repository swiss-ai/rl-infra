"""Expose BidirLM packed activations through CTFAlign's HF hidden-state API."""


class BidirLMHiddenStates:
    """Bypass the transformers-v4 model's broken intermediate-state unpacking.

    CTFAlign encodes one unpadded sequence per call. BidirLM prepends a BOS
    internally and packs the sequence; layer N's input is HF hidden_states[N].
    Capture that tensor, drop the internal BOS, and restore the batch dimension.
    Do not use upstream's hidden-state output, which indexes away the batch
    dimension and later slices the wrong end of the sequence.
    """

    def __init__(self, model, layer):
        self.model = model
        count = len(model.layers)
        self.layer = layer if layer >= 0 else count + 1 + layer
        if not 0 <= self.layer <= count:
            raise ValueError(f"BidirLM alignment layer must be in [0, {count}]")

    def __call__(self, input_ids, attention_mask=None, output_hidden_states=True, **kwargs):
        if input_ids.shape[0] != 1:
            raise ValueError("BidirLM CTFAlign adapter requires one sequence per call")
        if attention_mask is None or not bool(attention_mask.all()):
            raise ValueError("BidirLM CTFAlign adapter requires an unpadded attention mask")
        captured = []
        if self.layer == len(self.model.layers):
            handle = self.model.norm.register_forward_hook(
                lambda module, args, output: captured.append(output))
        else:
            handle = self.model.layers[self.layer].register_forward_pre_hook(
                lambda module, args: captured.append(args[0]))
        try:
            self.model(input_ids=input_ids, attention_mask=attention_mask,
                       output_hidden_states=False, **kwargs)
        finally:
            handle.remove()
        if len(captured) != 1 or captured[0].ndim != 2:
            raise RuntimeError("Unexpected BidirLM packed layer output")
        hidden = captured[0]
        if hidden.shape[0] != input_ids.shape[1] + 1:
            raise RuntimeError("BidirLM token count differs from input plus internal BOS")
        states = [None] * (len(self.model.layers) + 1)
        states[self.layer] = hidden[1:].unsqueeze(0)
        return {"hidden_states": tuple(states)}
