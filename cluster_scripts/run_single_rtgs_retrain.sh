#!/bin/sh
#SBATCH --time=04:00:00            # Request run time (wall-clock). Default is 1 minute
#SBATCH --nodes=1                  # Request 1 node
#SBATCH --tasks-per-node=1         # Set one task per node
#SBATCH --cpus-per-task=4          # Request number of CPUs (threads) per task.
#SBATCH --gres=gpu:a40:1               # Request 1 GPU
#SBATCH --mem=16GB                  # Request 4 GB of RAM in total
#SBATCH --output=slurm_output_rtgs_retrain/slurm-%x-%j.out   # Set name of output log. %j is the Slurm jobId
#SBATCH --error=slurm_output_rtgs_retrain/slurm-%x-%j.err    # Set name of error log. %j is the Slurm jobId


module use /opt/insy/modulefiles
module load cuda/11.8

SCENE="$1"
CAMERA="$2"
REP="$3" 

config_name=$(basename "$(find /tudelft.net/staff-umbrella/StudentsCVlab/jwarchocki/container_output/best_configs/rtgs/${CAMERA}/${SCENE} -maxdepth 1 -name "*.yaml" -print -quit)")
echo $config_name

mkdir -p /tudelft.net/staff-umbrella/StudentsCVlab/jwarchocki/container_output/rtgs/retrain/ego_exo/random_search/${CAMERA}/${SCENE}/${REP}

apptainer exec -C --nv \
	--bind /tudelft.net/staff-umbrella/StudentsCVlab/jwarchocki/container_output/rtgs/retrain/ego_exo/random_search/${CAMERA}/${SCENE}/${REP}:/opt/models/rtgs/output/ego_exo/random_search/${CAMERA}/${SCENE}:rw \
	--bind /tudelft.net/staff-umbrella/StudentsCVlab/jwarchocki/models/evaluation/output:/opt/models/rtgs/ego_exo_data:rw \
	--bind /tudelft.net/staff-umbrella/StudentsCVlab/jwarchocki/container_output/best_configs/rtgs/${CAMERA}/${SCENE}:/opt/models/rtgs/configs/ego_exo/random_configs:rw \
	daic_container.sif \
	/bin/bash -c $"source /opt/miniconda3/etc/profile.d/conda.sh && conda activate 4d_gaussian_splatting && cd /opt/models/rtgs && python train.py --config configs/ego_exo/random_configs/${config_name}"
