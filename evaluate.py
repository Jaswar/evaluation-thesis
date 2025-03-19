import os 
import numpy as np
import cv2 as cv
import json

def psnr(img1, img2, mask):
    img1 = np.reshape(img1, (-1, ))
    img2 = np.reshape(img2, (-1, ))
    mask = np.reshape(mask, (-1, ))
    mask = np.repeat(mask, 3)
    mask = mask > 0
    img1 = img1[mask]
    img2 = img2[mask]
    mse = ((img1 - img2) ** 2).mean()
    return 20 * np.log10(1.0 / np.sqrt(mse))



def evaluate(source_path, gt_path, renders_path):
    masks = [cv.imread(os.path.join(source_path, 'masks', f)) for f in sorted(os.listdir(os.path.join(source_path, 'masks')))]
    masks = [(mask[:, :, 0] > 0).astype(np.float32) for mask in masks]
    test_masks = masks[1::2]
    gts = [cv.imread(os.path.join(gt_path, f)) for f in sorted(os.listdir(gt_path))]
    gts = [gt / 255.0 for gt in gts]
    renders = [cv.imread(os.path.join(renders_path, f)) for f in sorted(os.listdir(renders_path))]
    renders = [render / 255.0 for render in renders]
    assert len(test_masks) == len(gts) == len(renders), f'Number of masks, gts and renders do not match: {len(test_masks)}, {len(gts)}, {len(renders)}'

    psnrs = []
    for i in range(len(test_masks)):
        psnr_ = psnr(gts[i], renders[i], test_masks[i])
        psnrs.append(psnr_)
    return np.mean(psnrs)


def get_paths_from_model(model, root_path, camera_label, scene):
    if model == '4DGaussians':
        source_path = os.path.join('output', camera_label, scene)
        gt_path = os.path.join(root_path, camera_label, scene, 'test/ours_14000/gt')
        renders_path = os.path.join(root_path, camera_label, scene, 'test/ours_14000/renders')
    elif model == 'Deformable-3D-Gaussians':
        source_path = os.path.join('output', camera_label, scene)
        gt_path = os.path.join(root_path, camera_label, scene, 'test/ours_40000/gt')
        renders_path = os.path.join(root_path, camera_label, scene, 'test/ours_40000/renders')
    elif model == '4d-gaussian-splatting':
        source_path = os.path.join('output', camera_label, scene)
        gt_path = os.path.join(root_path, camera_label, scene, 'test/ours_None/gt')
        renders_path = os.path.join(root_path, camera_label, scene, 'test/ours_None/renders')
    else:
        raise ValueError(f'Model not supported: {model}')
    return source_path, gt_path, renders_path


if __name__ == '__main__':
    models = ['Deformable-3D-Gaussians', '4DGaussians', '4d-gaussian-splatting']

    with open('settings.json', 'r') as f:
        settings = json.load(f)

    for model in models:
        root_path = f'../{model}/output/ego_exo'
        for setting in settings:
            scene = setting['take_name']
            for camera_label in ['camera-rgb', 'gopro']:
                source_path, gt_path, renders_path = get_paths_from_model(model, root_path, camera_label, scene)
                mean_psnr = evaluate(source_path, gt_path, renders_path)
                print(f'{model} {camera_label} {scene} PSNR: {mean_psnr:.2f}')
            
    