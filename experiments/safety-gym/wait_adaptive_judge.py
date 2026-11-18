import json
import os
import sys
import time
import urllib.error
import urllib.request

path, timeout_s = sys.argv[1], float(sys.argv[2])
expected_model = os.environ.get("SAFETY_ADAPTIVE_JUDGE_MODEL", f"{os.environ['USER']}/openai/gpt-oss-safeguard-120b")
expected_run_id = os.environ.get("SAFETY_RUN_ID")
deadline = time.monotonic() + timeout_s
last = "endpoint discovery file is not ready"
while time.monotonic() < deadline:
    try:
        with open(path, encoding="utf-8") as stream:
            payload = json.load(stream)
        if not isinstance(payload, dict):
            raise ValueError("endpoint discovery must be a JSON object")
        base_url, model, job_id = payload.get("base_url"), payload.get("model"), payload.get("job_id")
        if expected_run_id and payload.get("run_id") != expected_run_id:
            raise ValueError("endpoint file belongs to a different adaptive run")
        if not isinstance(job_id, str) or not job_id or model != expected_model:
            raise ValueError("endpoint job ID or served model does not match the expected SML judge")
        if not isinstance(base_url, str) or not base_url.startswith("http://") or not base_url.endswith(":8080/v1"):
            raise ValueError("SML judge must use a node-local http://host:8080/v1 endpoint")
        host = base_url[len("http://"):-len("/v1")]
        if not host or any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-:" for ch in host):
            raise ValueError("invalid node-local judge host")
        request = urllib.request.Request(base_url + "/models", headers={"Accept": "application/json"})
        with urllib.request.urlopen(request, timeout=5) as response:
            models = json.load(response)
        found = {entry.get("id") for entry in models.get("data", []) if isinstance(entry, dict)}
        if model not in found:
            raise ValueError("SML /v1/models does not advertise the expected judge model")
        print(base_url)
        print(model)
        raise SystemExit(0)
    except (OSError, ValueError, KeyError, urllib.error.URLError, json.JSONDecodeError) as error:
        last = str(error)
        time.sleep(5)
print(f"SML judge did not become ready: {last}", file=sys.stderr)
raise SystemExit(2)
