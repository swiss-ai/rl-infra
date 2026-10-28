#!/bin/bash
# Reads CUDA GPU core dumps (written on a device exception, CUDA_ENABLE_COREDUMP_ON_EXCEPTION=1)
# with cuda-gdb: the exception, the faulting kernel and PC, the device-side backtrace and the
# registers of every warp on the faulting SM. For a fault in UCCL's internode combine kernel it
# also reads the kernel's parameters (constant bank 0 from 0x210; 0x2b0 is d2h_channel_addrs,
# the table of queue handles, 0x2b8 their count) and, when the dump holds global memory, the
# table and the 32 handles it points to. Run inside the job's container (cuda-gdb ships with
# the image); no GPU needed. The report for <core> goes to <core>.txt.
#
# Usage: read_coredump.sh <core> [<core>...]
set -uo pipefail
for core in "$@"; do
    out="${core}.txt"
    warp_cmds=()
    for w in $(seq 0 63); do warp_cmds+=(-ex "echo \n===== warp ${w}\n" -ex "cuda warp ${w} lane 0" -ex "info registers"); done
    cuda-gdb -batch -nx \
        -ex "set pagination off" \
        -ex "target cudacore ${core}" \
        -ex "info cuda devices" \
        -ex "info cuda kernels" \
        -ex "info cuda warps" \
        -ex "bt" \
        -ex "x/12i \$pc-0x60" \
        -ex "p/x \$errorpc" \
        -ex "info registers" \
        -ex "echo \n===== kernel parameters (@parameter, 0x210-0x2c8)\n" -ex "x/24gx (@parameter unsigned long *) 0x210" \
        -ex "echo \n===== table pointer (@parameter 0x2b0)\n" -ex "x/1gx (@parameter unsigned long *) 0x2b0" \
        -ex "echo \n===== registers of all warps on the faulting SM\n" \
        "${warp_cmds[@]}" \
        > "${out}" 2>&1
    # d2h_channel_addrs is the kernel parameter at 0x2b0 (num_d2h_channel_addrs, 32, follows at 0x2b8).
    table=$(awk '/^===== table pointer/ {p=1; next} p && /:/ {sub(/^[^:]*:/, ""); print $1; exit}' "${out}")
    if [[ -n "${table}" && "${table}" != "0x0" ]]; then
        cuda-gdb -batch -nx \
            -ex "set pagination off" \
            -ex "target cudacore ${core}" \
            -ex "echo \n===== d2h_channel_addrs table at ${table} (32 entries)\n" \
            -ex "x/32gx (@global unsigned long *) ${table}" \
            -ex "echo \n===== same table through generic addressing (managed memory)\n" \
            -ex "x/32gx (@generic unsigned long *) ${table}" \
            -ex "echo \n===== handle objects at table - 0x600 (32 x 40 bytes, allocated just before the table)\n" \
            -ex "x/160gx (@global unsigned long *) (${table} - 0x600)" \
            -ex "x/160gx (@generic unsigned long *) (${table} - 0x600)" \
            -ex "echo \n===== memory around the table\n" \
            -ex "x/64gx (@global unsigned long *) (${table} - 0x100)" \
            -ex "echo \n===== handle objects the first entry points to (5 x 8 bytes each)\n" \
            -ex "x/160gx (@global unsigned long *) *(@global unsigned long *) ${table}" \
            >> "${out}" 2>&1
    else
        echo "table pointer not found in the parameter dump (table='${table}')" >> "${out}"
    fi
    echo "${out}"
done
