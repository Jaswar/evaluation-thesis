#!/bin/sh
#SBATCH --time=04:00:00            # Request run time (wall-clock). Default is 1 minut
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=16GB
#SBATCH --output=slurm_output_4dgs_renders/slurm-%x-%j.out
#SBATCH --error=slurm_output_4dgs_renders/slurm-%x-%j.err


module use /opt/insy/modulefiles
module load cuda/11.8

SCENES=("georgiatech_covid_03_4"
	"iiith_cooking_58_2"
	"unc_basketball_03-31-23_01_17"
	"minnesota_rockclimbing_013_2"
	"uniandes_basketball_003_43"
	"utokyo_pcr_2001_31_8"
	"georgiatech_bike_01_4"
	"indiana_bike_02_3"
	"iiith_cooking_111_2"
	"georgiatech_cooking_08_02_4")

# List of cameras
CAMERAS=("camera-rgb" "gopro")

# Loop through all scenes
for SCENE in "${SCENES[@]}"
do
    # Loop through all cameras
    for CAMERA in "${CAMERAS[@]}"
    do
	for REP in {0..2}; do
       	    echo "Rendering run_${SCENE}_${CAMERA}_${REP}"
    	    
	    config_name=$(basename "$(find /tudelft.net/staff-umbrella/StudentsCVlab/jwarchocki/container_output/best_configs/4DGaussians/${CAMERA}/${SCENE} -maxdepth 1 -name "*.py" -print -quit)")
	    echo $config_name
	    
	    apptainer exec -C --nv \
		--bind /tudelft.net/staff-umbrella/StudentsCVlab/jwarchocki/container_output/4DGaussians/retrain:/opt/models/4DGaussians-thesis/output:rw \
		--bind /tudelft.net/staff-umbrella/StudentsCVlab/jwarchocki/models/evaluation/output:/opt/models/4DGaussians-thesis/ego_exo_data:rw \
		--bind /tudelft.net/staff-umbrella/StudentsCVlab/jwarchocki/container_output/best_configs/4DGaussians/${CAMERA}/${SCENE}:/opt/models/4DGaussians-thesis/configs/ego_exo/random_configs:rw \
		daic_container.sif \
		/bin/bash -c $"source /opt/miniconda3/etc/profile.d/conda.sh && conda activate Gaussians4D && cd /opt/models/4DGaussians-thesis && python render.py -m output/output/ego_exo/random_search/${CAMERA}/${SCENE}/${REP} --configs configs/ego_exo/random_configs/${config_name} --eval"
	done
    done
done
