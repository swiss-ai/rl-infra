#!/bin/bash
# Runs a command (the NeMo-RL driver), then deletes this run's OpenSandbox sandboxes, and exits with the
# command's status. A run leaves sandboxes behind even when it succeeds (the batch in flight when training
# ends), and they would otherwise hold cluster capacity until their 6 h TTL. The launchers put this in front
# of the driver; a job killed by Slurm (scancel, time limit) never gets here, so reap those by hand
# (docs/harbor-osb-chonk-run.md, Cleanup).
#
#   reap_after.sh <command...>
#   env: OPENSANDBOX_DOMAIN, OPENSANDBOX_API_KEY, NEMO_GYM_RUN_ID, NEMO_GYM_USER (the launchers export them);
#        REAP_PYTHON, a Python that imports nemo_gym (default: the one apertus2/driver.sh runs, in the image)
"$@"
rc=$?
# Gym's own reaper (deletes only sandboxes labelled with this run and user). The key goes through the
# environment, not the command line.
"${REAP_PYTHON:-/usr/bin/python3.12}" -c '
import os, sys
from nemo_gym.sandbox.providers.opensandbox.cleanup_sandboxes import main
e = os.environ
sys.exit(main(["--domain", e["OPENSANDBOX_DOMAIN"], "--api-key", e["OPENSANDBOX_API_KEY"],
               "--run-id", e["NEMO_GYM_RUN_ID"], "--user", e["NEMO_GYM_USER"], "--reap"]))
' || echo "WARNING: sandbox reaping failed; reap run ${NEMO_GYM_RUN_ID:-?} by hand" >&2
exit ${rc}
