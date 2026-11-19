#!/bin/bash
# Chemistry rows for run 5 (envs/gen_chem_rdkit.py): one intTest node, CPU only, in the run's container.
# Submit from /scratch: cd /scratch/hanint06/data/stem-rl/chem_rdkit && sbatch <this file>
#SBATCH --job-name=gen-chem-rdkit
#SBATCH --partition=intTest
#SBATCH --account=in0138
#SBATCH --nodes=1
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --comment=field=nlp;appl=pytorch
#SBATCH --propagate=NONE
#SBATCH --output=/scratch/hanint06/data/stem-rl/chem_rdkit/gen.out
D=/scratch/hanint06/data/stem-rl/chem_rdkit
# RDKit 2025.09.5 (cp312, aarch64) from PyPI, unpacked next to the data; deleted after use (group file quota).
[[ -d "$D/pydeps/rdkit" ]] || python3 -m zipfile -e "$D/wheel/rdkit-2025.9.5-cp312-cp312-manylinux_2_28_aarch64.whl" "$D/pydeps"
srun --cpu-bind=none --cpus-per-task="${SLURM_CPUS_ON_NODE}" \
  --container-image=/scratch/hanint06/container-images/nemo-rl-apertus2-cfcfede4-ucclfix-cxisetup.sqsh \
  --container-mounts=/scratch/hanint06:/scratch/hanint06,/home01/hanint06:/home01/hanint06 \
  env CHEM_PYDEPS="$D/pydeps" python -I /home01/hanint06/projects/rl-infra-stem5/envs/gen_chem_rdkit.py \
  "$D/src/NCI-Open_2012-05-01.sdf.gz" "$D" --workers 64
