#!/bin/sh
#SBATCH --time=04:00:00            # Request run time (wall-clock). Default is 1 minute
#SBATCH --nodes=1                  # Request 1 node
#SBATCH --tasks-per-node=1         # Set one task per node
#SBATCH --cpus-per-task=4          # Request number of CPUs (threads) per task.
#SBATCH --gres=gpu:a40:1               # Request 1 GPU
#SBATCH --mem=16GB                  # Request 4 GB of RAM in total
#SBATCH --output=slurm_output_4dgs_retrain/slurm-%x-%j.out   # Set name of output log. %j is the Slurm jobId
#SBATCH --error=slurm_output_4dgs_retrain/slurm-%x-%j.err    # Set name of error log. %j is the Slurm jobId


module use /opt/insy/modulefiles
module load cuda/11.8

SCENE="$1"
CAMERA="$2"
REP="$3" 

config_name=$(basename "$(find /tudelft.net/staff-umbrella/StudentsCVlab/jwarchocki/container_output/best_configs/4DGaussians/${CAMERA}/${SCENE} -maxdepth 1 -name "*.py" -print -quit)")
echo $config_name

apptainer exec -C --nv \
	--bind /tudelft.net/staff-umbrella/StudentsCVlab/jwarchocki/container_output/4DGaussians/retrain:/opt/models/4DGaussians-thesis/output:rw \
	--bind /tudelft.net/staff-umbrella/StudentsCVlab/jwarchocki/models/evaluation/output:/opt/models/4DGaussians-thesis/ego_exo_data:rw \
	--bind /tudelft.net/staff-umbrella/StudentsCVlab/jwarchocki/container_output/best_configs/4DGaussians/${CAMERA}/${SCENE}:/opt/models/4DGaussians-thesis/arguments/ego_exo/random_configs:rw \
	daic_container.sif \
	/bin/bash -c $"source /opt/miniconda3/etc/profile.d/conda.sh && conda activate Gaussians4D && cd /opt/models/4DGaussians-thesis && python train.py -s ego_exo_data/all_saves/${CAMERA}/${SCENE} --port 6017 --expname output/ego_exo/random_search/${CAMERA}/${SCENE}/${REP} --configs arguments/ego_exo/random_configs/${config_name}"
