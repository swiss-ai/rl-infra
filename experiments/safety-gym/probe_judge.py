"""Check the selected local judge's advertised model without printing credentials."""
import json
import os
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

url = os.environ['SAFETY_JUDGE_BASE_URL'].rstrip('/')
if not url.endswith('/v1'):
    raise SystemExit('JUDGE_URL must end in /v1.')
key = 'EMPTY'
if os.environ.get('JUDGE_CONFIG'):
    from pathlib import Path
    try:
        import yaml
    except ImportError:
        import subprocess
        key = subprocess.run(['yq', 'r', os.environ['JUDGE_CONFIG'], 'judge_api_key'],
                             check=True, capture_output=True, text=True).stdout.strip()
    else:
        key = yaml.safe_load(Path(os.environ['JUDGE_CONFIG']).read_text())['judge_api_key']
request = Request(url + '/models', headers={'Authorization': 'Bearer ' + key})
wait_seconds = int(os.environ.get('SAFETY_JUDGE_WAIT_SECONDS', '0'))
if wait_seconds < 0:
    raise SystemExit('SAFETY_JUDGE_WAIT_SECONDS must be nonnegative.')
deadline = time.monotonic() + wait_seconds
announced_wait = False
while True:
    remaining = deadline - time.monotonic()
    try:
        timeout = min(10, max(0.1, remaining)) if wait_seconds else 10
        with urlopen(request, timeout=timeout) as response:
            ids = [model['id'] for model in json.load(response)['data']]
        break
    except HTTPError as error:
        if error.code != 503:
            raise SystemExit('Judge /models check failed: HTTP ' + str(error.code)) from None
        failure = 'HTTP 503'
    except (URLError, TimeoutError, ConnectionError) as error:
        failure = type(error).__name__
    if time.monotonic() >= deadline:
        raise SystemExit('Judge /models did not become ready within ' + str(wait_seconds) + ' seconds (last error: ' + failure + ').')
    if not announced_wait:
        print('Waiting up to ' + str(wait_seconds) + ' seconds for judge /models readiness.', flush=True)
        announced_wait = True
    time.sleep(min(10, max(0, deadline - time.monotonic())))
if os.environ['SAFETY_JUDGE_MODEL'] not in ids:
    raise SystemExit('The configured model is absent from judge /models.')
print('Judge /models ready:', os.environ['SAFETY_JUDGE_MODEL'])
