import os 
import numpy as np
import cv2 as cv
import json
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
from torch import Tensor
from lpips import LPIPS


import warnings
from typing import List, Optional, Tuple, Union

from common import get_masks, psnr, ssim, lpips, get_paths_from_model, get_mask_type_based_on_data_type
    

def evaluate(source_path, gt_path, renders_path, mask_type):
    if not os.path.exists(gt_path) or not os.path.exists(renders_path):
        return None, None, None

    masks = get_masks(source_path, mask_type)
    test_masks = masks[3::4]

    gts = [cv.imread(os.path.join(gt_path, f)) for f in sorted(os.listdir(gt_path)) if f.endswith('.png') or f.endswith('.jpg')]
    gts = [gt / 255.0 for gt in gts]
    renders = [cv.imread(os.path.join(renders_path, f)) for f in sorted(os.listdir(renders_path)) if f.endswith('.png') or f.endswith('.jpg')]
    renders = [render / 255.0 for render in renders]

    assert len(test_masks) == len(gts) == len(renders), f'Number of masks, gts and renders do not match: {len(test_masks)}, {len(gts)}, {len(renders)}'
    for (mask, gt, render) in zip(test_masks, gts, renders):
        assert mask.shape[0] == gt.shape[0] == render.shape[0], f'Height mismatch: {mask.shape[0]} vs {gt.shape[0]} vs {render.shape[0]}'
        assert mask.shape[1] == gt.shape[1] == render.shape[1], f'Width mismatch: {mask.shape[1]} vs {gt.shape[1]} vs {render.shape[1]}'
        assert gt.shape[2] == render.shape[2] == 3, f'Channel mismatch: {gt.shape[2]} vs {render.shape[2]}'

    # for (gt, render, mask) in zip(gts, renders, test_masks):
    #     cv.imshow('gt', gt)
    #     cv.imshow('render', render)
    #     cv.imshow('mask', mask)
    #     cv.waitKey(0)
    # cv.destroyAllWindows()

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
    return np.mean(psnrs), np.mean(ssims), np.mean(lpipss)

if __name__ == '__main__':
    models = ['EgoGaussian', 'Deformable-3D-Gaussians', '4DGaussians', '4d-gaussian-splatting']
    mask_type = 'full'
    selected_scenes = 'non_eg'  # eg or non_eg

    with open('settings.json', 'r') as f:
        settings = json.load(f)
    settings = [setting for setting in settings if setting['type'] == selected_scenes]

    results_over_repetitions = {}
    total_counts = {}
    for repetition in ['0', '1', '2']:
        results = {}
        for model in models:
            if model not in results:
                results[model] = {}
            # root_path = f'../{model}/output/ego_exo/with_val_set'
            root_path = f'retrain_output/{model}/retrain/ego_exo/random_search'
            for setting in settings[::-1]:
                scene = setting['take_name']
                scene_type = setting['type']
                if scene not in results[model]:
                    results[model][scene] = {}
                for camera_label in ['camera-rgb', 'gopro'] if model != 'EgoGaussian' else ['camera-rgb']:
                    if camera_label not in results[model][scene]:
                        results[model][scene][camera_label] = {}
                    source_path, gt_path, renders_path = get_paths_from_model(model, root_path, camera_label, scene, repetition)
                    task_specific_mask = mask_type
                    if camera_label == 'camera-rgb':
                        task_specific_mask = get_mask_type_based_on_data_type(task_specific_mask, scene_type)
                    mean_psnr, mean_ssim, mean_lpips = evaluate(source_path, gt_path, renders_path, task_specific_mask)
                    print(f'{model} {camera_label} {scene} {repetition} PSNR: {mean_psnr} SSIM: {mean_ssim} LPIPS: {mean_lpips}')
                    results[model][scene][camera_label] = {'psnr': mean_psnr, 'ssim': mean_ssim, 'lpips': mean_lpips}
        print()

        for model in models:
            if model == 'EgoGaussian':
                continue
            if model not in total_counts:
                total_counts[model] = {}
            for scene in results[model]:
                for metric in ['psnr', 'ssim', 'lpips']:
                    if metric not in total_counts[model]:
                        total_counts[model][metric] = 0
                    if metric != 'lpips' and results[model][scene]['gopro'][metric] > results[model][scene]['camera-rgb'][metric]:
                        total_counts[model][metric] += 1
                    elif metric == 'lpips' and results[model][scene]['gopro'][metric] < results[model][scene]['camera-rgb'][metric]:
                        total_counts[model][metric] += 1

        mean_over_scenes = {}
        for model in results:
            mean_over_scenes[model] = {}
            for scene in results[model]:
                for camera_label in results[model][scene]:
                    if camera_label not in mean_over_scenes[model]:
                        mean_over_scenes[model][camera_label] = {}
                    for metric in results[model][scene][camera_label]:
                        if metric not in mean_over_scenes[model][camera_label]:
                            mean_over_scenes[model][camera_label][metric] = []
                        if results[model][scene][camera_label][metric] is not None:
                            mean_over_scenes[model][camera_label][metric].append(results[model][scene][camera_label][metric])
            for camera_label in mean_over_scenes[model]:
                for metric in mean_over_scenes[model][camera_label]:
                    mean = np.mean(mean_over_scenes[model][camera_label][metric])
                    if model not in results_over_repetitions:
                        results_over_repetitions[model] = {}
                    if camera_label not in results_over_repetitions[model]:
                        results_over_repetitions[model][camera_label] = {}
                    if metric not in results_over_repetitions[model][camera_label]:
                        results_over_repetitions[model][camera_label][metric] = []
                    results_over_repetitions[model][camera_label][metric].append(mean)
    print(total_counts)
    model_mapping = {
        'EgoGaussian': 'EgoGaussian',
        'Deformable-3D-Gaussians': 'Def3DGS',
        '4DGaussians': '4DGS',
        '4d-gaussian-splatting': 'RTGS',
    }
    camera_mapping = {
        'camera-rgb': 'Ego',
        'gopro': 'Exo',
    }
    result = ''
    for model in results_over_repetitions:
        result += f'{model_mapping[model]} '
        for camera_label in results_over_repetitions[model]:
            result += f'& {camera_mapping[camera_label]} '
            for metric in results_over_repetitions[model][camera_label]:
                res = results_over_repetitions[model][camera_label][metric]
                mean = np.mean(res)
                std = np.std(res)
                if metric == 'psnr':
                    result += f'& {mean:.2f} $\\pm$ {std:.2f} '
                else:
                    result += f'& {mean:.2f} $\\pm$ {std:.3f} '  # higher resolution std
            result += '\\\\\n'
    print(result)

    result = ''