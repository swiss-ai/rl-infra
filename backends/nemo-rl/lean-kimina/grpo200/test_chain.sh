#!/bin/bash
# Local test of start.sh, guard.sh, stop.sh with stand-ins for sbatch, squeue, sacct, scancel and run-200.sh, on a
# fake scratch tree in a temporary folder. Runs anywhere (no Slurm); touches nothing else.
set -uo pipefail
SRC="$(cd "$(dirname "$0")" && pwd)"
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
S="$T/scratch"; C="$S/lean/grpo200"; B="$T/bin"
mkdir -p "$C" "$B" "$S/rl-infra-245e760/backends/nemo-rl/lean-kimina"
cp "$SRC"/{settings.sh,start.sh,guard.sh,stop.sh,status.sh} "$C/"
echo 100 > "$T/id"
next_id='n=$(cat "$T/id"); echo $((n+1)) > "$T/id"; echo $n'
cat > "$B/sbatch" <<EOF
#!/bin/bash
T="$T"
if [[ " \$* " == *" --test-only "* ]]; then echo "sbatch: Job 1 to start at \${STUB_EST:-2099-01-01T00:00:00} a using 3456 processors" >&2; exit 0; fi
echo "\$*" >> "\$T/sbatch.log"; $next_id
EOF
cat > "$B/squeue" <<EOF
#!/bin/bash
T="$T"
if [[ " \$* " == *" -j "* ]]; then id=\$(echo "\$*" | sed -E 's/.*-j ([0-9]+).*/\1/'); grep "^\$id " "\$T/states" 2>/dev/null | awk '{print \$2}'; else cat "\$T/active" 2>/dev/null; fi
exit 0
EOF
cat > "$B/sacct" <<EOF
#!/bin/bash
T="$T"
id=\$(echo "\$*" | sed -E 's/.*-j ([0-9]+).*/\1/'); grep "^\$id|" "\$T/sacct" 2>/dev/null | cut -d'|' -f2-
exit 0
EOF
cat > "$B/scancel" <<EOF
#!/bin/bash
echo "\$*" >> "$T/scancel.log"
EOF
R200="$S/rl-infra-245e760/backends/nemo-rl/lean-kimina/run-200.sh"
cat > "$R200" <<EOF
#!/bin/bash
T="$T"
echo "RUN=\$RUN PARTITION=\$PARTITION JOB_TIME=\$JOB_TIME DEPENDENCY=\$DEPENDENCY SBATCH_EXTRA=\$SBATCH_EXTRA OUT=\$OUT LEAN_PROJECT=\$LEAN_PROJECT LEAN_TOOLCHAIN=\$LEAN_TOOLCHAIN DRY_RUN=\$DRY_RUN" >> "\$T/run200.log"
[[ "\$DRY_RUN" == 1 ]] && { echo "== DRY_RUN: would run"; exit 0; }
id=\$($next_id); echo "Run \$RUN is job \$id."
EOF
chmod +x "$B"/*
export PATH="$B:$PATH" SCRATCH="$S" SCRATCH_ROOT="$S" USER=tester
fails=0
check() { if eval "$1"; then echo "ok   $2"; else echo "FAIL $2"; fails=$((fails+1)); fi; }
reset() { rm -f "$T"/{sbatch.log,run200.log,scancel.log,active,states,sacct} "$C"/{chain.tsv,guard.tsv,DONE,STOP}; rm -rf "$S/nemo-rl-grpo200"; }
R="$S/nemo-rl-grpo200/results/lean-kimina-v02-grpo200-128x8"
guard() { env CHAIN_DIR="$C" CHAIN_K="$1" TRAIN_JOB="$2" bash "$C/guard.sh" > "$T/guard.out" 2>&1; echo $?; }
row() { awk -F'\t' -v k="$1" -v c="$2" '$1==k {print $c}' "$C/chain.tsv"; }

# Dry run: nothing submitted, run-200.sh once in dry-run mode, preemptable when normal starts late.
reset; out=$(cd "$C" && DRY_RUN=1 bash start.sh 2>&1); rc=$?; [[ -n "${DEBUG:-}" ]] && echo "$out"
check '[[ $rc == 0 && ! -e $T/sbatch.log && ! -e $C/chain.tsv ]] && grep -q "dry run: nothing submitted" <<< "$out"' "dry run submits nothing"
check '[[ $(wc -l < $T/run200.log) -eq 1 ]] && grep -q "DRY_RUN=1" $T/run200.log && grep -q "PARTITION=preemptable JOB_TIME=24:00:00" $T/run200.log' "dry run: one run-200.sh dry run, preemptable 24 h"
check 'grep -q "G4: sbatch" <<< "$out" && ! grep -q "G5" <<< "$out"' "dry run plans 4 jobs on preemptable"
reset; soon=$(python3 -c 'import datetime as d; print((d.datetime.now()+d.timedelta(minutes=20)).strftime("%H:%M:%S"))')
out=$(cd "$C" && STUB_EST=$soon DRY_RUN=1 bash start.sh 2>&1)
check 'grep -q "PARTITION=normal JOB_TIME=12:00:00" $T/run200.log && grep -q "G5: sbatch" <<< "$out"' "normal (12 h, 5 jobs) when it starts within the hour (time-of-day estimate)"

# Real start: the chain's dependencies, settings and records.
reset; out=$(cd "$C" && bash start.sh 2>&1); rc=$?
check '[[ $rc == 0 && $(grep -vc "^#" $C/chain.tsv) == 4 ]]' "4 training jobs + 4 guards submitted"
J1=$(row 1 2) G1=$(row 1 3) J2=$(row 2 2) G2=$(row 2 3) J3=$(row 3 2) G3=$(row 3 3) J4=$(row 4 2) G4=$(row 4 3)
check 'sed -n 1p $T/run200.log | grep -q "DEPENDENCY= SBATCH_EXTRA=--kill-on-invalid-dep=yes"' "J1 has no dependency, kill-on-invalid-dep"
check 'sed -n 2p $T/run200.log | grep -q "DEPENDENCY=afterok:${G1},singleton"' "J2 waits for G1 (afterok, singleton)"
check 'grep -q "dependency=afterany:${J1} " $T/sbatch.log && grep -q "partition=xfer" $T/sbatch.log && grep -q "CHAIN_K=4,TRAIN_JOB=${J4}" $T/sbatch.log' "guards wait for their training job (afterany) on xfer"
check 'grep -q "OUT=$S/nemo-rl-grpo200 LEAN_PROJECT=/capstor/store/cscs/swissai/infra01/lean_apertus_v2/lean-v4.34.0 LEAN_TOOLCHAIN=/capstor/store/cscs/swissai/infra01/lean_apertus_v2/lean-v4.34.0/elan/toolchains/leanprover--lean4---v4.34.0" $T/run200.log' "separate OUT and capstor Lean reach run-200.sh"
check '[[ $(grep -c "RUN=lean-kimina-v02-grpo200-128x8 " $T/run200.log) == 4 ]]' "same RUN for every job"
echo "$J1" > "$T/active"; out=$(cd "$C" && bash start.sh 2>&1); rc=$?
check '[[ $rc == 1 ]] && grep -q "still in the queue" <<< "$out"' "refuses a second chain while jobs are queued"
rm -f "$T/active"

# Guards. J1 ended without a checkpoint: forgiven once.
echo "$J1|TIMEOUT|1-00:00:00|2|0:0" > "$T/sacct"
rc=$(guard 1 "$J1"); check '[[ $rc == 0 ]] && grep -q "continue-no-progress" $C/guard.tsv' "G1: first start without progress continues"
echo "$J2|FAILED|00:20:00|0|1:0" >> "$T/sacct"
rc=$(guard 2 "$J2"); check '[[ $rc == 1 ]] && grep -q "stop-no-progress" $C/guard.tsv' "G2: second time without progress stops"
check 'grep -q "^$J3 $G3 $J4 $G4$" $T/scancel.log' "G2 cancels J3 G3 J4 G4"
# A new chain with progress.
reset; (cd "$C" && bash start.sh > /dev/null 2>&1)
J1=$(row 1 2) J2=$(row 2 2) J3=$(row 3 2) G3=$(row 3 3) J4=$(row 4 2) G4=$(row 4 3)
mkdir -p "$R/step_10" "$R/tmp_step_20"
rc=$(guard 1 "$J1"); check '[[ $rc == 0 ]] && tail -1 $C/guard.tsv | grep -q "	10	continue	"' "G1: step 10 (tmp_step_20 ignored) continues"
mkdir -p "$R/step_110" "$R/old_step_130"
rc=$(guard 2 "$J2"); check '[[ $rc == 0 ]] && tail -1 $C/guard.tsv | grep -q "	110	continue	"' "G2: step 110 (old_step_130 ignored) continues"
printf '%s PENDING\n' "$J2" > "$T/states"
rc=$(guard 2 "$J2"); check '[[ $rc == 0 ]] && tail -1 $C/guard.tsv | grep -q "PENDING.*continue-alive"' "G2: training job still queued (requeued) lets the chain go on"
mkdir -p "$R/step_200"
rc=$(guard 3 "$J3"); check '[[ $rc == 1 && -e $C/DONE ]] && tail -1 $C/guard.tsv | grep -q "	200	done	"' "G3: step 200 is done, writes DONE"
check 'grep -q "^$J2 $J4 $G4$" $T/scancel.log' "G3 cancels J4 G4 and the still-queued J2"
rm -f "$T/states"
out=$(cd "$C" && bash start.sh 2>&1); check '[[ $? == 1 ]] && grep -q DONE <<< "$out"' "start.sh refuses after DONE"
# STOP file and the chain's end.
reset; (cd "$C" && bash start.sh > /dev/null 2>&1); J1=$(row 1 2) J4=$(row 4 2)
mkdir -p "$R/step_20"; touch "$C/STOP"
rc=$(guard 1 "$J1"); check '[[ $rc == 1 ]] && tail -1 $C/guard.tsv | grep -q "stop-file"' "STOP file stops the chain"
out=$(cd "$C" && bash start.sh 2>&1); check '[[ $? == 1 ]] && grep -q STOP <<< "$out"' "start.sh refuses while STOP exists"
rm -f "$C/STOP"; mkdir -p "$R/step_60"
rc=$(guard 4 "$J4"); check '[[ $rc == 1 ]] && tail -1 $C/guard.tsv | grep -q "chain-exhausted" && grep -q "start.sh again" $T/guard.out' "last guard with progress: chain exhausted, says to start again"
# stop.sh
reset; (cd "$C" && bash start.sh > /dev/null 2>&1); J1=$(row 1 2) G1=$(row 1 3) J2=$(row 2 2)
printf '%s RUNNING\n%s PENDING\n%s PENDING\n' "$J1" "$G1" "$J2" > "$T/states"
out=$(bash "$C/stop.sh" 2>&1); check '[[ ! -e $C/STOP && ! -e $T/scancel.log ]] && grep -q "3 job(s) would be cancelled" <<< "$out"' "stop.sh without --yes changes nothing"
out=$(bash "$C/stop.sh" --yes 2>&1); check '[[ -e $C/STOP ]] && grep -q "^$J1 $G1 $J2$" $T/scancel.log' "stop.sh --yes writes STOP and cancels the live jobs"
# Guards against the wrong scratch.
out=$(SCRATCH="$T" bash "$C/status.sh" 2>&1); check '[[ $? == 1 ]] && grep -q "is not" <<< "$out"' "refuses when SCRATCH is not SCRATCH_ROOT"
cp "$C/start.sh" "$T/start.sh"; cp "$C/settings.sh" "$T/settings.sh"
out=$(bash "$T/start.sh" 2>&1); check '[[ $? == 1 ]] && grep -q "run start.sh from" <<< "$out"' "start.sh runs only from the chain folder"
echo "$fails failure(s)"; [[ $fails -eq 0 ]]
