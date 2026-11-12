"""Reference-based MetricX-24 scoring, isolated for model-backed checks."""

from pathlib import Path

from .scoring import bounded


class MetricXReward:
    def __init__(self, config):
        import torch
        from transformers import AutoTokenizer
        from .models import load_module

        self.torch = torch
        self.device = config.get("device", "cuda:0")
        self.max_length = int(config.get("max_length", 1536))
        if not 3 <= self.max_length <= 1536:
            raise ValueError("MetricX-24 max_length must be between 3 and its trained limit of 1536")
        upstream = load_module(Path(config["walar_dir"]) / "code/models.py",
                               "mt_metricx_upstream")
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.get("metricx_tokenizer", "google/mt5-xl"))
        self.model, self.loading_info = upstream.MT5ForRegression.from_pretrained(
            config.get("metricx_model", "google/metricx-24-hybrid-xxl-v2p6-bfloat16"),
            torch_dtype=torch.bfloat16, output_loading_info=True)
        # A wrong model class/checkpoint must not leave random scoring weights.
        for key in ("missing_keys", "mismatched_keys", "error_msgs"):
            if self.loading_info.get(key):
                raise RuntimeError(f"MetricX checkpoint did not load completely: {key}={self.loading_info[key]}")
        self.model.to(self.device).eval()

    def score(self, source, hypothesis, reference):
        if not isinstance(reference, str) or not reference.strip():
            raise ValueError("Reference-based MetricX requires a nonempty reference")
        # Match google-research/metricx metricx24/predict.py's non-QE branch.
        inputs = self.tokenizer(
            f"source: {source} candidate: {hypothesis} reference: {reference}",
            max_length=self.max_length, truncation=True, padding=False,
            return_tensors="pt")
        inputs = {k: v[:, :-1].to(self.device) for k, v in inputs.items()}
        with self.torch.inference_mode():
            error = float(self.model(**inputs).predictions.reshape(-1)[0])
        return -bounded(error, 0, 25, "MetricX error")
