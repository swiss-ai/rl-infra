"""JSON-lines inference server; execute using the isolated COMET Python environment."""

import argparse
from contextlib import redirect_stdout
import ctypes
import json
import os
from pathlib import Path
import signal
import sys
import traceback


def move_to_device(value, device, torch):
    """Preserve COMET's nested prediction batch (dict or tuple of dicts)."""
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: move_to_device(item, device, torch) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device, torch) for item in value)
    if isinstance(value, list):
        return [move_to_device(item, device, torch) for item in value]
    return value


def score_translation(model, row, device, torch):
    batch = model.prepare_sample([{"src": row["src"], "mt": row["mt"]}], stage="predict")
    batch = move_to_device(batch, device, torch)
    # Kiwi's UnifiedMetric returns a tuple of input sequences. predict_step
    # handles that structure and any model-specific score aggregation; plain
    # referenceless regression models instead accept a dict here.
    prediction = model.predict_step(batch, batch_idx=0)
    return float(prediction.scores.reshape(-1)[0])


def main():
    # Ray kills the owning reward actor at shutdown. Ensure its GPU child exits
    # too on Linux, including when the actor is terminated without destructors.
    if sys.platform == "linux":
        parent = os.getppid()
        if ctypes.CDLL(None, use_errno=True).prctl(1, signal.SIGTERM) != 0:
            raise OSError(ctypes.get_errno(), "Cannot set COMET parent-death signal")
        if os.getppid() != parent:
            return
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", required=True)
    args = parser.parse_args()
    with redirect_stdout(sys.stderr):
        import torch
        from comet import download_model, load_from_checkpoint
        checkpoint = args.model if Path(args.model).is_file() else download_model(args.model)
        model = load_from_checkpoint(checkpoint).to(args.device).eval()
        if model.requires_references():
            raise ValueError("COMET-Kiwi checkpoint must be reference-free")
    print(json.dumps({"ready": True}), flush=True)
    for line in sys.stdin:
        try:
            row = json.loads(line)
            with redirect_stdout(sys.stderr), torch.inference_mode():
                score = score_translation(model, row, args.device, torch)
            print(json.dumps({"score": score}, allow_nan=False), flush=True)
        except Exception as error:
            traceback.print_exc(file=sys.stderr)
            print(json.dumps({"error": str(error)}), flush=True)
            return


if __name__ == "__main__":
    main()
