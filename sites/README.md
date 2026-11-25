# sites

One profile per cluster: account, partitions, GPUs per node, filesystems, container runtime and image locations.
`load.sh` loads the configuration; the first file that sets a variable wins:

1. the environment at launch (`SIZE=ultra bash run.sh stem`)
2. `site.env` in the repository root (personal, gitignored; copy `site.env.example`)
3. `sites/<SITE>.env`
4. `backends/nemo-rl/env/stack.env`

| File | |
|---|---|
| `alps.env` | CSCS Alps (Clariden): 4 GH200 per node, CSCS container engine (EDF) |
| `hangang.env` | KISTI Hangang (pyxis/enroot); compute ended 2026-11-23, kept for reference |
| `load.sh` | `source sites/load.sh && rl_infra_load` |

The site is detected from the filesystems. A new cluster needs `sites/<name>.env`, and `SITE=<name>` if its
filesystems do not identify it.
