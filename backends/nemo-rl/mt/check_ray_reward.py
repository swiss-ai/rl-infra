"""Check the actual NeMo-RL MT environment and GPU reward actor on one node."""

import argparse
import json
import os
from pathlib import Path


def main():
    import ray

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Scorer JSON configuration")
    parser.add_argument("--scorer-python", required=True)
    parser.add_argument("--timeout", type=float, default=900)
    args = parser.parse_args()
    package_dir = str(Path(__file__).resolve().parent)
    os.environ["PYTHONPATH"] = package_dir + os.pathsep + os.environ.get("PYTHONPATH", "")
    from mt_reward.environment import TranslationEnvironment

    # This is a standalone single-node check, not a connection to another job.
    ray.init(num_cpus=8, num_gpus=1, include_dashboard=False)
    environment = None
    try:
        environment = TranslationEnvironment.options(runtime_env={
            "env_vars": {"PYTHONPATH": os.environ["PYTHONPATH"]}
        }).remote({
            "scorer_config": str(Path(args.config).resolve()),
            "scorer_python": args.scorer_python,
            "resources": {"num_nodes": 1, "gpus_per_node": 1, "num_cpus": 4},
            "timeout": args.timeout,
        })
        source = "El gato está durmiendo en el sofá junto a la ventana."
        reference = "Die Katze schläft auf dem Sofa neben dem Fenster."
        messages = [[{"role": "assistant", "content": text}]
                    for text in (reference, source)]
        info = {"source_lang": "es", "target_lang": "de", "source": source,
                "reference": reference}
        result = ray.get(environment.step.remote(messages, [info.copy(), info.copy()]),
                         timeout=args.timeout)
        scores = [json.loads(row["content"])["mt_reward"] for row in result.observations]
        for score in scores:
            print(json.dumps(score), flush=True)
        if result.rewards.shape != (2,) or not result.terminateds.all().item():
            raise RuntimeError("Invalid NeMo-RL reward/termination batch")
        if not scores[0]["language_alignment"] or scores[1]["reward"] != -25:
            raise RuntimeError("Unexpected language-gate results through Ray")
        ray.get(environment.shutdown.remote(), timeout=60)
        print("Ray MT reward check passed (training not started)")
    finally:
        if environment is not None:
            ray.kill(environment)
        ray.shutdown()


if __name__ == "__main__":
    main()
