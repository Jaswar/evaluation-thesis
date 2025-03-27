import os 
import numpy as np
import cv2 as cv
import json

def psnr(img1, img2, mask):
    if np.sum(mask) == 0:
        return None
    img1 = np.reshape(img1, (-1, ))
    img2 = np.reshape(img2, (-1, ))
    mask = np.reshape(mask, (-1, ))
    mask = np.repeat(mask, 3)
    mask = mask > 0
    img1 = img1[mask]
    img2 = img2[mask]
    mse = ((img1 - img2) ** 2).mean()
    return 20 * np.log10(1.0 / np.sqrt(mse))


def get_full_masks(source_path, ordering):
    masks = []
    for i, cam_name in enumerate(ordering):
        mask_path = os.path.join(source_path, cam_name, 'masks', f'{i:05d}.png')
        mask = cv.imread(mask_path)
        mask = (mask[:, :, 0] > 0).astype(np.float32)
        masks.append(mask)
    return masks


def obj_is_dynamic(inx, ranges):
    for r in ranges:
        if r[0] <= inx < r[1]:
            return True
    return False


def combine_dynamic_masks(source_path, ordering):
    with open(os.path.join(source_path, 'dynamics.json'), 'r') as f:
        dynamics = json.load(f)
    masks = []
    height, width = cv.imread(os.path.join(source_path, ordering[0], 'masks', '00000.png')).shape[:2]
    for i, cam_name in enumerate(ordering):
        dynamic_mask = np.zeros((height, width), dtype=np.bool)
        for obj in dynamics:
            id_ = obj['id']
            ranges = obj['ranges']
            mask_path = os.path.join(source_path, cam_name, 'dynamic_masks', str(id_), f'{i:05d}.png')
            if obj_is_dynamic(i, ranges) and os.path.exists(mask_path):
                mask = cv.imread(mask_path)
                assert mask.shape[0] == height and mask.shape[1] == width, f'Mask shape mismatch: {mask.shape} vs {height}x{width}'
                mask = mask[:, :, 0] > 0
                dynamic_mask |= mask
        masks.append(dynamic_mask)
    return masks


def get_dynamic_masks(source_path, ordering):
    masks = combine_dynamic_masks(source_path, ordering)
    combined_masks = []
    for i, (mask, cam_name) in enumerate(zip(masks, ordering)):
        cam_mask = cv.imread(os.path.join(source_path, cam_name, 'masks', f'{i:05d}.png'))
        cam_mask = cam_mask[:, :, 0] > 0
        combined_mask = np.logical_and(mask, cam_mask)
        combined_masks.append(combined_mask.astype(np.float32))
    return combined_masks


def get_static_masks(source_path, ordering):
    masks = combine_dynamic_masks(source_path, ordering)
    combined_masks = []
    for i, (mask, cam_name) in enumerate(zip(masks, ordering)):
        cam_mask = cv.imread(os.path.join(source_path, cam_name, 'masks', f'{i:05d}.png'))
        cam_mask = cam_mask[:, :, 0] > 0
        combined_mask = np.logical_and(1 - mask, cam_mask)
        combined_masks.append(combined_mask.astype(np.float32))
    return combined_masks


def get_masks(source_path, mask_type):
    with open(os.path.join(source_path, 'ordering.txt'), 'r') as f:
        ordering = f.read().split('\n')
    ordering = [o for o in ordering if o != '']

    if mask_type == 'full':
        masks = get_full_masks(source_path, ordering)
    elif mask_type == 'dynamic':
        masks = get_dynamic_masks(source_path, ordering)
    elif mask_type == 'static':
        masks = get_static_masks(source_path, ordering)
    else:
        raise ValueError(f'Mask type not supported: {mask_type}')
    return masks
    

def evaluate(source_path, gt_path, renders_path, mask_type):
    if not os.path.exists(gt_path) or not os.path.exists(renders_path):
        return 0

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
    for i in range(len(test_masks)):
        psnr_ = psnr(gts[i], renders[i], test_masks[i])
        if psnr_ is not None:
            psnrs.append(psnr_)
    return np.mean(psnrs)


def get_paths_from_model(model, root_path, camera_label, scene):
    if model == '4DGaussians':
        source_path = os.path.join('output/all_saves', camera_label, scene)
        gt_path = os.path.join(root_path, camera_label, scene, 'test/ours_14000/gt')
        renders_path = os.path.join(root_path, camera_label, scene, 'test/ours_14000/renders')
    elif model == 'Deformable-3D-Gaussians':
        source_path = os.path.join('output/all_saves', camera_label, scene)
        gt_path = os.path.join(root_path, camera_label, scene, 'test/ours_40000/gt')
        renders_path = os.path.join(root_path, camera_label, scene, 'test/ours_40000/renders')
    elif model == '4d-gaussian-splatting':
        source_path = os.path.join('output/all_saves', camera_label, scene)
        gt_path = os.path.join(root_path, camera_label, scene, 'test/ours_None/gt')
        renders_path = os.path.join(root_path, camera_label, scene, 'test/ours_None/renders')
    elif model == 'EgoGaussian':
        source_path = os.path.join('output/all_saves', camera_label, scene)
        gt_path = os.path.join('../EgoGaussian/output/HOI4D', scene, 'full/evaluation/all/gt')
        renders_path = os.path.join('../EgoGaussian/output/HOI4D', scene, 'full/evaluation/all/render')
    else:
        raise ValueError(f'Model not supported: {model}')
    return source_path, gt_path, renders_path


if __name__ == '__main__':
    models = ['EgoGaussian', 'Deformable-3D-Gaussians', '4DGaussians', '4d-gaussian-splatting']
    mask_type = 'full'

    with open('settings.json', 'r') as f:
        settings = json.load(f)

    for model in models:
        root_path = f'../{model}/output/ego_exo/with_val_set'
        for setting in settings[::-1]:
            scene = setting['take_name']
            for camera_label in ['camera-rgb', 'gopro'] if model != 'EgoGaussian' else ['camera-rgb']:
                source_path, gt_path, renders_path = get_paths_from_model(model, root_path, camera_label, scene)
                mean_psnr = evaluate(source_path, gt_path, renders_path, mask_type)
                print(f'{model} {camera_label} {scene} PSNR: {mean_psnr:.2f}')
            
    