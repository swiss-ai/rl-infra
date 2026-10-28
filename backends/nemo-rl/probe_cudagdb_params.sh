#!/bin/bash
# One-off probe: which cuda-gdb syntax reads a kernel's parameter (constant bank 0) space in a
# core dump. Tries the candidate address-space spellings at the combine kernel's table pointer
# (offset 0x2b0, or 0xa0 counted from the first parameter at 0x210) and prints cuda-gdb's help
# on the topic. Output goes to <core>.probe.txt.
#
# Usage: probe_cudagdb_params.sh <core>
set -uo pipefail
core=$1
cmds=(-ex "set pagination off" -ex "target cudacore ${core}" -ex "show version")
for space in parameter param_space params constant const_space constbank cbank global generic local; do
    for off in 0x210 0x2b0 0xa0 0x0; do
        cmds+=(-ex "echo \n===== @${space} ${off}\n" -ex "x/2gx (@${space} unsigned long *) ${off}")
    done
done
for c in 'info cuda kernels' 'info cuda launch trace' 'info frame' 'info args' 'p $c0' 'p $cbank' 'help info cuda' 'apropos address space' 'apropos param' 'help set cuda'; do
    cmds+=(-ex "echo \n===== ${c}\n" -ex "${c}")
done
cuda-gdb -batch -nx "${cmds[@]}" > "${core}.probe.txt" 2>&1
echo "${core}.probe.txt"
