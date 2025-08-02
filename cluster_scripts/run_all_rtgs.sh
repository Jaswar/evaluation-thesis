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
	echo "Running run_${SCENE}_${CAMERA}"
        # Submit the job with dynamic job name based on scene and camera
        sbatch --job-name="run_${SCENE}_${CAMERA}" run_single_rtgs.sh "$SCENE" "$CAMERA"
	sleep 5
    done
done
