"""Derive an experiment container from the unchanged rl-infra template."""
import json
import os
from pathlib import Path
import subprocess
import sys
import shutil

from adaptive_overlay import render_native_client_redaction, render_native_context_measure
from compat import render_adaptive_collector, render_adaptive_grpo

here = Path(__file__).resolve().parent
runtime = Path(sys.argv[1])
runtime.mkdir(parents=True, exist_ok=True)
source = Path(os.environ['NEMO_RL_DIR']) / 'nemo_rl/environments/nemo_gym.py'
adapter = runtime / 'nemo_gym.py'
subprocess.run([sys.executable, str(here / 'compat.py'), '--source', str(source), '--output', str(adapter)], check=True)
feature_profile = os.environ.get('SAFETY_PROFILE', 'legacy') != 'legacy'
feature_gym_adapter = None
feature_runtime_adapters = []
feature_models_config = None
feature_client_adapter = None
if feature_profile:
    secret_source = Path(os.environ['SAFETY_SWISSAI_ENV_FILE'])
    if not secret_source.is_file() or (secret_source.stat().st_mode & 0o777) != 0o600:
        raise RuntimeError('SwissAI private env file must exist with mode 0600')
    if secret_source.stat().st_uid != os.getuid():
        raise RuntimeError('SwissAI private env file must be owned by the submitting user')
    os.environ['SAFETY_SWISSAI_ENV_FILE'] = '/run/secrets/swissai.env'
    gym_source = Path(os.environ['SAFETY_GYM_DIR']) / 'responses_api_models/vllm_model/app.py'
    feature_gym_adapter = runtime / 'gym-vllm-model-app.py'
    feature_gym_adapter.write_text(render_native_context_measure(gym_source.read_text()))
    client_source = Path(os.environ["SAFETY_GYM_DIR"]) / "nemo_gym/client/openai_client.py"
    feature_client_adapter = runtime / "gym-openai-client.py"
    feature_client_adapter.write_text(render_native_client_redaction(client_source.read_text()))
    feature_models_config = runtime / "adaptive-models.yaml"
    shutil.copyfile(here / "adaptive_models.yaml", feature_models_config)
    os.environ["SAFETY_ADAPTIVE_MODELS_CONFIG"] = "/run/adaptive-models.yaml"
    for relative, renderer, target in (
        ('nemo_rl/algorithms/grpo.py', render_adaptive_grpo, '/opt/nemo-rl/nemo_rl/algorithms/grpo.py'),
        ('nemo_rl/algorithms/async_utils/trajectory_collector.py', render_adaptive_collector,
         '/opt/nemo-rl/nemo_rl/algorithms/async_utils/trajectory_collector.py'),
    ):
        original = Path(os.environ['NEMO_RL_DIR']) / relative
        generated = runtime / Path(relative).name
        generated.write_text(renderer(original.read_text()))
        feature_runtime_adapters.append((str(generated), target))
template = (here / '../../backends/nemo-rl/apertus2/edf.toml.template').read_text()
marker = '  "@@NEMO_RL_DIR@@:/opt/nemo-rl",\n'
assert template.count(marker) == 1
mounts = [
    (os.environ['SAFETY_GYM_DIR'], '/opt/nemo-rl/3rdparty/Gym-workspace/Gym'),
    (str(adapter), '/opt/nemo-rl/nemo_rl/environments/nemo_gym.py'),
    (os.environ['SAFETY_UV_BIN'] + '/uv', '/usr/local/bin/uv'),
    (os.environ['SAFETY_UV_BIN'] + '/uvx', '/usr/local/bin/uvx'),
    # Bind the metadata overlay and shared weights at their absolute paths, read-only.
    (os.environ['SAFETY_CHONK_MODEL'], os.environ['SAFETY_CHONK_MODEL'] + ':ro'),
    (os.environ['SAFETY_CHONK_SOURCE'], os.environ['SAFETY_CHONK_SOURCE'] + ':ro'),
]
if feature_gym_adapter is not None:
    mounts.append((str(feature_gym_adapter), '/opt/nemo-rl/3rdparty/Gym-workspace/Gym/responses_api_models/vllm_model/app.py'))
    mounts.extend(feature_runtime_adapters)
    mounts.append((str(secret_source), '/run/secrets/swissai.env:ro'))
    mounts.append((str(feature_models_config), '/run/adaptive-models.yaml:ro'))
    mounts.append((str(here / 'wait_adaptive_judge.py'), '/run/wait_adaptive_judge.py:ro'))
    mounts.append((str(feature_client_adapter), '/opt/nemo-rl/3rdparty/Gym-workspace/Gym/nemo_gym/client/openai_client.py'))
template = template.replace(marker, marker + ''.join('  ' + json.dumps(a + ':' + b) + ',\n' for a, b in mounts))
template = template.replace('PYTHONPATH = "', 'PYTHONPATH = "' + str(here) + ':', 1)
if feature_profile:
    gym_site_packages = Path(os.environ['SAFETY_GYM_SITE_PACKAGES'])
    if not gym_site_packages.is_dir():
        raise RuntimeError(f'Prepared Gym site-packages are missing: {gym_site_packages}')
    gym_pythonpath_anchor = '@@PYDEPS@@"\nPYDEPS ='
    if template.count(gym_pythonpath_anchor) != 1:
        raise RuntimeError('Unknown feature container PYTHONPATH template')
    template = template.replace(
        gym_pythonpath_anchor,
        '@@PYDEPS@@:' + str(gym_site_packages) + '"\nPYDEPS =',
        1,
    )
# This is a local vLLM service. Authenticated deployments may provide an existing
# private Gym config path; no key is copied to the rendered container environment.
secret = os.environ.get('JUDGE_CONFIG')
if secret:
    os.environ['SAFETY_JUDGE_CONFIG'] = secret
else:
    secret_path = runtime / 'judge.yaml'
    secret_path.write_text('judge_api_key: EMPTY\n')
    secret_path.chmod(0o600)
    os.environ['SAFETY_JUDGE_CONFIG'] = str(secret_path)
names = [name for name in os.environ if name.startswith('SAFETY_')]
names += ['JOB_UV_CACHE_DIR', 'HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE']
if os.environ.get('JUDGE_CONFIG'):
    names.append('JUDGE_CONFIG')
os.environ['HF_HUB_OFFLINE'] = os.environ['TRANSFORMERS_OFFLINE'] = '1'
template += '\n' + '\n'.join(name + ' = ' + json.dumps(os.environ[name]) for name in sorted(names)) + '\n'
(runtime / 'container.toml').write_text(template)
print('Runtime overlay:', runtime)
