import os
import json
import numpy as np

from common import get_masks, psnr, ssim, lpips, get_paths_from_model, get_mask_type_based_on_data_type

import matplotlib.pyplot as plt
import cv2 as cv
import scipy
import blur_detector
from tqdm import tqdm


def get_blurs(source_path, gt_path, mask_type):
    masks = get_masks(source_path, mask_type)
    test_masks = masks[3::4]
    gts = [cv.imread(os.path.join(gt_path, f)) for f in sorted(os.listdir(gt_path))]
    blur_scores = []
    for gt, mask in tqdm(zip(gts, test_masks)):
        gray = cv.cvtColor(gt, cv.COLOR_BGR2GRAY)
        # laplacian = cv.Laplacian(gray, cv.CV_64F)
        # blur_score = laplacian[mask == 1].var()
        blur_map = blur_detector.detectBlur(gray, downsampling_factor=4, num_scales=4, scale_start=2, num_iterations_RF_filter=3, show_progress=False)
        blur_map = 1 - blur_map  # originally, 1 represents sharp objects
        blur_score = blur_map[mask == 1].mean()
        blur_scores.append(blur_score)
    return blur_scores


def evaluate(source_path, gt_path, renders_path, mask_type):
    if not os.path.exists(gt_path) or not os.path.exists(renders_path):
        return None, None, None

    masks = get_masks(source_path, mask_type)
    test_masks = masks[3::4]

    gts = [cv.imread(os.path.join(gt_path, f)) for f in sorted(os.listdir(gt_path))]
    gts = [gt / 255.0 for gt in gts]
    renders = [cv.imread(os.path.join(renders_path, f)) for f in sorted(os.listdir(renders_path))]
    renders = [render / 255.0 for render in renders]

    assert len(test_masks) == len(gts) == len(renders), f'Number of masks, gts and renders do not match: {len(test_masks)}, {len(gts)}, {len(renders)}'
    for (mask, gt, render) in zip(test_masks, gts, renders):
        assert mask.shape[0] == gt.shape[0] == render.shape[0], f'Height mismatch: {mask.shape[0]} vs {gt.shape[0]} vs {render.shape[0]}'
        assert mask.shape[1] == gt.shape[1] == render.shape[1], f'Width mismatch: {mask.shape[1]} vs {gt.shape[1]} vs {render.shape[1]}'
        assert gt.shape[2] == render.shape[2] == 3, f'Channel mismatch: {gt.shape[2]} vs {render.shape[2]}'
    
    psnrs = []
    ssims = []
    lpipss = []
    for i in range(len(test_masks)):
        psnr_ = psnr(gts[i], renders[i], test_masks[i])
        ssim_ = ssim(gts[i], renders[i], test_masks[i])
        lpips_ = lpips(gts[i], renders[i], test_masks[i])
        if psnr_ is not None:
            psnrs.append(psnr_)
        if ssim_ is not None:
            ssims.append(ssim_)
        if lpips_ is not None:
            lpipss.append(lpips_)
    return psnrs, ssims, lpipss


def normalize(arr, make_log=True):
    if make_log:
        arr = np.log(arr)
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))

def main():
    models = ['EgoGaussian', 'Deformable-3D-Gaussians', '4DGaussians', '4d-gaussian-splatting']
    model_mapping = {
        'EgoGaussian': 'EgoGaussian',
        'Deformable-3D-Gaussians': 'Def3DGS',
        '4DGaussians': '4DGS',
        '4d-gaussian-splatting': 'RTGS',
    }

    with open('settings.json', 'r') as f:
        settings = json.load(f)

    plt.rcParams.update({'font.size': 22})
    scene_blurs = {}
    for model in models:
        all_blurs = []
        all_lpipss = []
        for repetition in ['0', '1', '2']:
            root_path = f'retrain_output/{model}/retrain/ego_exo/random_search'
            for setting in settings[::-1]:
                scene = setting['take_name']
                print(f'Processing {repetition=}, {model=}, {scene=}')
                scene_type = setting['type']
                if model == 'EgoGaussian' and scene_type == 'non_eg':
                    continue
                source_path, gt_path, renders_path = get_paths_from_model(model, root_path, 'camera-rgb', scene, repetition)
                task_specific_mask = get_mask_type_based_on_data_type('static', scene_type)

                if scene not in scene_blurs:
                    scene_blurs[scene] = get_blurs(source_path, gt_path, task_specific_mask)
                blur_scores = scene_blurs[scene]

                psnrs, ssims, lpipss = evaluate(source_path, gt_path, renders_path, task_specific_mask)
                lpipss = normalize(lpipss, make_log=False)
                blur_scores = normalize(blur_scores, make_log=False)
                all_blurs.extend(blur_scores)
                all_lpipss.extend(lpipss)
        print(f'Spearman statistics for blur: {scipy.stats.spearmanr(all_blurs, all_lpipss)}')
        print(f'Pearson statistics for blur: {scipy.stats.pearsonr(all_blurs, all_lpipss)}')
        plt.scatter(all_blurs, all_lpipss, label=model_mapping[model])
    plt.legend()
    plt.xlabel('Blurriness')
    plt.ylabel('LPIPS')
    plt.show()
    

if __name__ == '__main__':
    main()