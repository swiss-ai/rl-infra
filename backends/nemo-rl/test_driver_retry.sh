#!/bin/bash
# Local test of driver_retry.sh's save deadline (checkpointing.checkpoint_must_save_by) with a stand-in driver and
# `ray`. Runs anywhere (no Slurm, no Ray); touches only a temporary folder.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
mkdir -p "$T/bin" "$T/logs" "$T/ckpt"
# The stand-in driver records its arguments; FAIL_FIRST=1 makes the first attempt write a step record and fail.
cat > "$T/driver" <<EOF
#!/bin/bash
echo "\$*" >> "$T/args"
n=\$(wc -l < "$T/args")
if [[ "\${FAIL_FIRST:-0}" == 1 && \$n -eq 1 ]]; then touch "$T/logs/train_data_step1.jsonl"; exit 3; fi
exit 0
EOF
printf '#!/bin/bash\necho "0.0/48.0 GPU"\n' > "$T/bin/ray"
chmod +x "$T/driver" "$T/bin/ray"
export PATH="$T/bin:$PATH"
KEY=checkpointing.checkpoint_must_save_by
fails=0
check() { if eval "$1"; then echo "ok   $2"; else echo "FAIL $2"; fails=$((fails+1)); fi; }
run() { rm -f "$T/args"; env "$@" bash "$HERE/driver_retry.sh" "$T/driver" --config x.yaml "logger.log_dir=$T/logs" \
          "checkpointing.checkpoint_dir=$T/ckpt" ${ARGS[@]+"${ARGS[@]}"} > "$T/out" 2>&1; }
# Seconds of a DD:HH:MM:SS value of KEY in line $1 of the recorded arguments.
deadline() { sed -n "${1}p" "$T/args" | tr ' ' '\n' | grep "^${KEY}=" | cut -d= -f2 \
             | awk -F: '{print $1*86400 + $2*3600 + $3*60 + $4}'; }
near() { local v; v=$(deadline "$1"); [[ -n "$v" ]] && (( v >= $2 - 60 && v <= $2 + 5 )); }

ARGS=("${KEY}=00:23:30:00")
run NRL_JOB_END_EPOCH=
check '[[ $(deadline 1) == 84600 ]] && ! grep -q "job ends" $T/out' "without NRL_JOB_END_EPOCH the submitted deadline stays"
run NRL_JOB_END_EPOCH=$(( $(date +%s) + 8 * 3600 ))
check 'near 1 $(( 8 * 3600 - 1800 )) && [[ $(grep -o "${KEY}=" $T/args | wc -l) -eq 1 ]]' "an 8 h job: 7.5 h, replacing the submitted value"
run NRL_JOB_END_EPOCH=$(( $(date +%s) + 8 * 3600 )) NRL_SAVE_MARGIN_S=3600
check 'near 1 $(( 7 * 3600 ))' "NRL_SAVE_MARGIN_S sets the margin"
run NRL_JOB_END_EPOCH=$(( $(date +%s) + 600 ))
check '[[ $(deadline 1) == 300 ]]' "an end closer than the margin gives 5 min, not a negative deadline"
ARGS=()
run NRL_JOB_END_EPOCH=$(( $(date +%s) + 26 * 3600 ))
check 'near 1 $(( 26 * 3600 - 1800 )) && sed -n 1p $T/args | grep -q "${KEY}=01:01:30:00\|${KEY}=01:01:29:"' "added when the command has none (DD:HH:MM:SS with days)"
ARGS=("${KEY}=00:23:30:00")
run NRL_JOB_END_EPOCH=$(( $(date +%s) + 8 * 3600 )) FAIL_FIRST=1 DRIVER_RETRY_MIN_S=60
check '[[ $(wc -l < $T/args) -eq 2 ]] && near 2 $(( 8 * 3600 - 1800 )) && grep -q "attempt 1 started" $T/out' "a retry gets the time left before the job's end"
run NRL_JOB_END_EPOCH=$(( $(date +%s) + 1800 + 900 )) FAIL_FIRST=1 DRIVER_RETRY_MIN_S=1800
check '[[ $(wc -l < $T/args) -eq 1 ]] && grep -q "not retrying" $T/out' "no retry when less than DRIVER_RETRY_MIN_S is left"
echo "$fails failure(s)"; [[ $fails -eq 0 ]]
