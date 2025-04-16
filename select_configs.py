# only run on the cluster!

import os 

def render(camera_label, scene, config):
    config_name = config.replace('model', 'config') + '.py'
    os.system(f"""cd containers && apptainer exec -C --nv \
        --bind /tudelft.net/staff-umbrella/StudentsCVlab/jwarchocki/container_output/4DGaussians:/opt/models/4DGaussians-thesis/output:rw \\
        --bind /tudelft.net/staff-umbrella/StudentsCVlab/jwarchocki/models/evaluation/output:/opt/models/4DGaussians-thesis/ego_exo_data:rw \\
        --bind /tudelft.net/staff-umbrella/StudentsCVlab/jwarchocki/container_output/4DGaussians/configs/{camera_label}/{scene}:/opt/models/4DGaussians-thesis/arguments/ego_exo/random_configs:rw \\
        daic_container.sif \\
        /bin/bash -c $\"source /opt/miniconda3/etc/profile.d/conda.sh && conda activate Gaussians4D && cd /opt/models/4DGaussians-thesis && python render.py -m output/output/ego_exo/random_search/{camera_label}/{scene}/{config} --configs arguments/ego_exo/random_configs/{config_name} --eval\"""")

def copy_files(out_path, model_path, config_path):
    os.makedirs(out_path, exist_ok=True)
    os.system(f"cp -r {os.path.join(model_path, 'test')} {out_path}")
    os.system(f"cp -r {os.path.join(model_path, 'train')} {out_path}")
    os.system(f"cp -r {config_path} {out_path}")


def main(root_path, model, out_base_path):
    os.makedirs(out_base_path, exist_ok=True)

    output_path = os.path.join(root_path, model, 'output/ego_exo/random_search')
    for camera_model in ['camera-rgb', 'gopro']:
        in_camera_path = os.path.join(output_path, camera_model)
        for scene in sorted(os.listdir(in_camera_path)):
            in_scene_path = os.path.join(in_camera_path, scene)
            max_psnr = 0.0
            best_config = ''
            for config in sorted(os.listdir(in_scene_path)):
                psnr_test_path = os.path.join(in_scene_path, config, 'psnr_test.txt')
                with open(psnr_test_path, 'r') as f:
                    lines = f.read().split('\n')
                lines = [line for line in lines if line != '']
                final_psnr = float(lines[-1].split(' ')[1])
                if final_psnr > max_psnr:
                    max_psnr = final_psnr
                    best_config = config
            print(f'{camera_model} {scene}: PSNR={max_psnr}, best_config={best_config}')

            render(camera_model, scene, best_config)
            best_config_name = best_config.replace('model', 'config') + '.py'
            copy_files(os.path.join(out_base_path, camera_model, scene), 
                       os.path.join(output_path, camera_model, scene, best_config),
                       os.path.join(root_path, model, 'configs', camera_model, scene, best_config_name))

if __name__ == '__main__':
    root_path = 'container_output'
    model = '4DGaussians'
    out_base_path = f'container_output/best_configs/{model}'
    main(root_path, model, out_base_path)