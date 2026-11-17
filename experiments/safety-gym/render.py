"""Derive an experiment container from the unchanged rl-infra template."""
import json
import os
from pathlib import Path
import subprocess
import sys

here = Path(__file__).resolve().parent
runtime = Path(sys.argv[1])
runtime.mkdir(parents=True, exist_ok=True)
source = Path(os.environ['NEMO_RL_DIR']) / 'nemo_rl/environments/nemo_gym.py'
adapter = runtime / 'nemo_gym.py'
subprocess.run([sys.executable, str(here / 'compat.py'), '--source', str(source), '--output', str(adapter)], check=True)
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
template = template.replace(marker, marker + ''.join('  ' + json.dumps(a + ':' + b) + ',\n' for a, b in mounts))
template = template.replace('PYTHONPATH = "', 'PYTHONPATH = "' + str(here) + ':', 1)
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
