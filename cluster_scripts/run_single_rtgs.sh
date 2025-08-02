#!/bin/sh
#SBATCH --time=04:00:00            # Request run time (wall-clock). Default is 1 minute
#SBATCH --nodes=1                  # Request 1 node
#SBATCH --tasks-per-node=1         # Set one task per node
#SBATCH --cpus-per-task=4          # Request number of CPUs (threads) per task.
#SBATCH --gres=gpu:a40:1               # Request 1 GPU
#SBATCH --mem=16GB                  # Request 4 GB of RAM in total
#SBATCH --output=slurm_output_rtgs/slurm-%x-%j.out   # Set name of output log. %j is the Slurm jobId
#SBATCH --error=slurm_output_rtgs/slurm-%x-%j.err    # Set name of error log. %j is the Slurm jobId


module use /opt/insy/modulefiles
module load cuda/11.8

SCENE="$1"
CAMERA="$2"

mkdir "/tudelft.net/staff-umbrella/StudentsCVlab/jwarchocki/container_output/rtgs/configs/${CAMERA}"
mkdir "/tudelft.net/staff-umbrella/StudentsCVlab/jwarchocki/container_output/rtgs/configs/${CAMERA}/${SCENE}"

apptainer exec -C --nv \
	--bind /tudelft.net/staff-umbrella/StudentsCVlab/jwarchocki/container_output/rtgs:/opt/models/rtgs/output:rw \
	--bind /tudelft.net/staff-umbrella/StudentsCVlab/jwarchocki/models/evaluation/output:/opt/models/rtgs/ego_exo_data:rw \
	--bind /tudelft.net/staff-umbrella/StudentsCVlab/jwarchocki/container_output/rtgs/configs/${CAMERA}/${SCENE}:/opt/models/rtgs/configs/ego_exo/random_configs:rw \
	daic_container.sif \
	/bin/bash -c $"source /opt/miniconda3/etc/profile.d/conda.sh && conda activate 4d_gaussian_splatting && cd /opt/models/rtgs && python random_search.py --data_path ego_exo_data/all_saves/${CAMERA}/${SCENE} --model_path output/ego_exo/random_search/${CAMERA}/${SCENE} --ignore_errors"
