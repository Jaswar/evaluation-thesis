import cv2 as cv
import numpy as np
import json
import os

def combine_dynamic_masks(source_path, ordering, skip_ids):
    with open(os.path.join(source_path, 'dynamics.json'), 'r') as f:
        dynamics = json.load(f)
    masks = []
    height, width = cv.imread(os.path.join(source_path, ordering[0], 'masks', '00000.png')).shape[:2]
    for i, cam_name in enumerate(ordering):
        dynamic_mask = np.zeros((height, width), dtype=np.bool)
        for obj in dynamics:
            id_ = obj['id']
            if id_ in skip_ids:
                continue
            ranges = obj['ranges']
            mask_path = os.path.join(source_path, cam_name, 'dynamic_masks', str(id_), f'{i:05d}.png')
            if obj_is_dynamic(i, ranges) and os.path.exists(mask_path):
                mask = cv.imread(mask_path)
                assert mask.shape[0] == height and mask.shape[1] == width, f'Mask shape mismatch: {mask.shape} vs {height}x{width}'
                mask = mask[:, :, 0] > 0
                dynamic_mask |= mask
        masks.append(dynamic_mask)
    return masks

def get_dynamic_masks(source_path, ordering, skip_ids=None):
    if skip_ids is None:
        skip_ids = []
    masks = combine_dynamic_masks(source_path, ordering, skip_ids)
    combined_masks = []
    for i, (mask, cam_name) in enumerate(zip(masks, ordering)):
        cam_mask = cv.imread(os.path.join(source_path, cam_name, 'masks', f'{i:05d}.png'))
        cam_mask = cam_mask[:, :, 0] > 0
        combined_mask = np.logical_and(mask, cam_mask)
        combined_masks.append(combined_mask.astype(np.float32))
    return combined_masks

def get_full_masks(source_path, ordering):
    masks = []
    for i, cam_name in enumerate(ordering):
        mask_path = os.path.join(source_path, cam_name, 'masks', f'{i:05d}.png')
        mask = cv.imread(mask_path)
        mask = (mask[:, :, 0] > 0).astype(np.float32)
        masks.append(mask)
    return masks

def get_masks(source_path, frame_idx):
    with open(os.path.join(source_path, 'ordering.txt'), 'r') as f:
        ordering = f.read().split('\n')
    ordering = [o for o in ordering if o != '']
    return get_full_masks(source_path, ordering)[frame_idx * 4 + 3], get_dynamic_masks(source_path, ordering)[frame_idx * 4 + 3]

def obj_is_dynamic(inx, ranges):
    for r in ranges:
        if r[0] <= inx < r[1]:
            return True
    return False

def load(source_path, gts_path, renders_path, frame_idx):
    gt_path = sorted(os.listdir(gts_path))[frame_idx]
    render_path = sorted(os.listdir(renders_path))[frame_idx]
    
    gt = cv.imread(os.path.join(gts_path, gt_path))
    render = cv.imread(os.path.join(renders_path, render_path))
    mask, dynamic_mask = get_masks(source_path, frame_idx)
    mask = mask[:, :, None]
    dynamic_mask = dynamic_mask[:, :, None]

    gt = gt * mask
    dynamic_mask = np.concatenate((np.zeros_like(dynamic_mask, dtype=np.float32), dynamic_mask, np.zeros_like(dynamic_mask, dtype=np.float32)), axis=2) * 255
    gt = gt * 0.8 + dynamic_mask * 0.2

    return gt, render * mask


def get_paths_from_model(model, root_path, camera_label, scene, repetition):
    if model == '4DGaussians':
        source_path = os.path.join('output/all_saves', camera_label, scene)
        iteration = [d for d in os.listdir(os.path.join(root_path, camera_label, scene, repetition, 'test')) if d.startswith('ours_')][0]
        gt_path = os.path.join(root_path, camera_label, scene, f'{repetition}/test/{iteration}/gt')
        renders_path = os.path.join(root_path, camera_label, scene, f'{repetition}/test/{iteration}/renders')
    elif model == 'Deformable-3D-Gaussians':
        source_path = os.path.join('output/all_saves', camera_label, scene)
        iteration = [d for d in os.listdir(os.path.join(root_path, camera_label, scene, repetition, 'test')) if d.startswith('ours_')][0]
        gt_path = os.path.join(root_path, camera_label, scene, repetition, f'test/{iteration}/gt')
        renders_path = os.path.join(root_path, camera_label, scene, repetition, f'test/{iteration}/renders')
    elif model == '4d-gaussian-splatting':
        source_path = os.path.join('output/all_saves', camera_label, scene)
        inner_path = os.listdir(os.path.join(root_path, camera_label, scene, repetition))[0]
        gt_path = os.path.join(root_path, camera_label, scene, repetition, inner_path, 'test/ours_None/gt')
        renders_path = os.path.join(root_path, camera_label, scene, repetition, inner_path, 'test/ours_None/renders')
    elif model == 'EgoGaussian':
        source_path = os.path.join('output/all_saves', camera_label, scene)
        gt_path = os.path.join('../EgoGaussian/output/HOI4D', f'{scene}_{repetition}', 'full/evaluation/all/gt')
        renders_path = os.path.join('../EgoGaussian/output/HOI4D', f'{scene}_{repetition}', 'full/evaluation/all/render')
    else:
        raise ValueError(f'Model not supported: {model}')
    return source_path, gt_path, renders_path


if __name__ == '__main__':
    models = ['Deformable-3D-Gaussians', '4DGaussians', '4d-gaussian-splatting']
    scene = 'unc_basketball_03-31-23_01_17'
    repetition = 0
    frame_idx = 1
    # models = ['EgoGaussian', 'Deformable-3D-Gaussians', '4DGaussians', '4d-gaussian-splatting']
    # scene = 'iiith_cooking_111_2'
    # repetition = 0
    # frame_idx = 0
    out_path = 'vis_output'

    with open('settings.json', 'r') as f:
        settings = json.load(f)
    repetition = str(repetition)

    for model in models:
        # root_path = f'../{model}/output/ego_exo/with_val_set'
        root_path = f'retrain_output/{model}/retrain/ego_exo/random_search'            
        for camera_label in ['camera-rgb', 'gopro'] if model != 'EgoGaussian' else ['camera-rgb']:
            source_path, gt_path, renders_path = get_paths_from_model(model, root_path, camera_label, scene, repetition)
            gt, render = load(source_path, gt_path, renders_path, frame_idx)
            cv.imwrite(os.path.join(out_path, f'gt_{model}_{camera_label}.png'), gt)
            cv.imwrite(os.path.join(out_path, f'render_{model}_{camera_label}.png'), render)