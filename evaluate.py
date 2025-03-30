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

# Copyright 2020 by Gongfan Fang, Zhejiang University.
# All rights reserved.
import warnings
from typing import List, Optional, Tuple, Union


lpips_fn = LPIPS(net='vgg', spatial=True).cuda()
def lpips(img1, img2, mask):
    if np.sum(mask) == 0:
        return None
    img1 = torch.tensor(img1).permute(2, 0, 1).unsqueeze(0).float().cuda()
    img2 = torch.tensor(img2).permute(2, 0, 1).unsqueeze(0).float().cuda()
    mask = torch.tensor(mask).unsqueeze(0).unsqueeze(0).float().cuda()
    _lpips = lpips_fn(img1 * 2 - 1, img2 * 2 - 1)
    _lpips = _lpips[mask == 1].mean()
    return _lpips.cpu().item()


def _fspecial_gauss_1d(size: int, sigma: float) -> Tensor:
    r"""Create 1-D gauss kernel
    Args:
        size (int): the size of gauss kernel
        sigma (float): sigma of normal distribution
    Returns:
        torch.Tensor: 1D kernel (1 x 1 x size)
    """
    coords = torch.arange(size, dtype=torch.float)
    coords -= size // 2

    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()

    return g.unsqueeze(0).unsqueeze(0)

def gaussian_filter(input: Tensor, win: Tensor) -> Tensor:
    r""" Blur input with 1-D kernel
    Args:
        input (torch.Tensor): a batch of tensors to be blurred
        window (torch.Tensor): 1-D gauss kernel
    Returns:
        torch.Tensor: blurred tensors
    """
    assert all([ws == 1 for ws in win.shape[1:-1]]), win.shape
    if len(input.shape) == 4:
        conv = F.conv2d
    elif len(input.shape) == 5:
        conv = F.conv3d
    else:
        raise NotImplementedError(input.shape)

    C = input.shape[1]
    out = input
    for i, s in enumerate(input.shape[2:]):
        if s >= win.shape[-1]:
            out = conv(out, weight=win.transpose(2 + i, -1), stride=1, padding=0, groups=C)
        else:
            warnings.warn(
                f"Skipping Gaussian Smoothing at dimension 2+{i} for input: {input.shape} and win size: {win.shape[-1]}"
            )

    return out

def _ssim(
    X: Tensor,
    Y: Tensor,
    data_range: float,
    win: Tensor,
    size_average: bool = True,
    K: Union[Tuple[float, float], List[float]] = (0.01, 0.03),
    mask: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    r""" Calculate ssim index for X and Y

    Args:
        X (torch.Tensor): images
        Y (torch.Tensor): images
        data_range (float or int): value range of input images. (usually 1.0 or 255)
        win (torch.Tensor): 1-D gauss kernel
        size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
        mask (torch.Tensor): boolean mask same size as X and Y

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: ssim results.
    """
    K1, K2 = K
    # batch, channel, [depth,] height, width = X.shape
    compensation = 1.0

    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    win = win.to(X.device, dtype=X.dtype)

    mu1 = gaussian_filter(X, win)
    mu2 = gaussian_filter(Y, win)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = compensation * (gaussian_filter(X * X, win) - mu1_sq)
    sigma2_sq = compensation * (gaussian_filter(Y * Y, win) - mu2_sq)
    sigma12 = compensation * (gaussian_filter(X * Y, win) - mu1_mu2)

    cs_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)  # set alpha=beta=gamma=1
    ssim_map = ((2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)) * cs_map

    if mask is not None:
        # Interpolate the mask to match the size of ssim_map and cs_map
        mask_resized = F.interpolate(mask, size=ssim_map.shape[-2:], mode='bilinear')

        # Flatten the ssim_map, cs_map, and resized mask
        ssim_map_flat = torch.flatten(ssim_map, 2)
        cs_map_flat = torch.flatten(cs_map, 2)
        mask_flat = torch.flatten(mask_resized, 2)

        # Apply the mask to the flattened ssim_map and cs_map
        masked_ssim_map = torch.masked_select(ssim_map_flat, mask_flat.bool())
        masked_cs_map = torch.masked_select(cs_map_flat, mask_flat.bool())

        # Compute the mean of the masked ssim_map and cs_map
        masked_ssim_per_channel = masked_ssim_map.mean(-1)
        masked_cs = masked_cs_map.mean(-1)

        return masked_ssim_per_channel, masked_cs
    ssim_per_channel = torch.flatten(ssim_map, 2).mean(-1)
    cs = torch.flatten(cs_map, 2).mean(-1)
    return ssim_per_channel, cs


def ssim_torch(
    X: Tensor,
    Y: Tensor,
    data_range: float = 1.0,
    size_average: bool = True,
    win_size: int = 11,
    win_sigma: float = 1.5,
    win: Optional[Tensor] = None,
    K: Union[Tuple[float, float], List[float]] = (0.01, 0.03),
    nonnegative_ssim: bool = False,
    mask: Optional[Tensor] = None,
) -> Tensor:
    r""" interface of ssim
    Args:
        X (torch.Tensor): a batch of images, (N,C,H,W)
        Y (torch.Tensor): a batch of images, (N,C,H,W)
        data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
        size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
        win_size: (int, optional): the size of gauss kernel
        win_sigma: (float, optional): sigma of normal distribution
        win (torch.Tensor, optional): 1-D gauss kernel. if None, a new kernel will be created according to win_size and win_sigma
        K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
        nonnegative_ssim (bool, optional): force the ssim response to be nonnegative with relu
        mask (torch.Tensor): boolean mask same size as X and Y

    Returns:
        torch.Tensor: ssim results
    """
    if not X.shape == Y.shape:
        raise ValueError(f"Input images should have the same dimensions, but got {X.shape} and {Y.shape}.")

    if mask is not None and mask.shape != X.shape:
        raise ValueError(f"Input mask should have the same dimensions as input images, but got {mask.shape} and {X.shape}.")

    for d in range(len(X.shape) - 1, 1, -1):
        X = X.squeeze(dim=d)
        Y = Y.squeeze(dim=d)
        if mask is not None:
            mask = mask.squeeze(dim=d)

    #if mask is not None:
    #    assert size_average is True, "per channel ssim is not available if mask exist"
    #    margin = win_size // 2
    #    mask = mask[..., margin:-margin, margin:-margin]

    if len(X.shape) not in (4, 5):
        raise ValueError(f"Input images should be 4-d or 5-d tensors, but got {X.shape}")

    #if not X.type() == Y.type():
    #    raise ValueError(f"Input images should have the same dtype, but got {X.type()} and {Y.type()}.")

    if win is not None:  # set win_size
        win_size = win.shape[-1]

    if not (win_size % 2 == 1):
        raise ValueError("Window size should be odd.")

    if win is None:
        win = _fspecial_gauss_1d(win_size, win_sigma)
        win = win.repeat([X.shape[1]] + [1] * (len(X.shape) - 1))

    ssim_per_channel, cs = _ssim(X, Y, data_range=data_range, win=win, size_average=False, K=K, mask=mask)
    if nonnegative_ssim:
        ssim_per_channel = torch.relu(ssim_per_channel)

    #if mask is not None:
    #    return ssim_per_channel
    #elif size_average:
    if size_average:
        return ssim_per_channel.mean()
    else:
        return ssim_per_channel.mean(1)


def ssim(img1, img2, mask):
    if np.sum(mask) == 0:
        return None
    img1 = torch.tensor(img1).permute(2, 0, 1).unsqueeze(0).float()
    img2 = torch.tensor(img2).permute(2, 0, 1).unsqueeze(0).float()
    mask = torch.tensor(mask).unsqueeze(0).repeat(3, 1, 1).unsqueeze(0).float()
    return ssim_torch(img1, img2, mask=mask).item()


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


def get_static_masks(source_path, ordering):
    masks = combine_dynamic_masks(source_path, ordering, skip_ids=[])
    combined_masks = []
    for i, (mask, cam_name) in enumerate(zip(masks, ordering)):
        cam_mask = cv.imread(os.path.join(source_path, cam_name, 'masks', f'{i:05d}.png'))
        cam_mask = cam_mask[:, :, 0] > 0
        combined_mask = np.logical_and(1 - mask, cam_mask)
        combined_masks.append(combined_mask.astype(np.float32))
    return combined_masks


def get_full_masks_no_hand(source_path, ordering):
    masks = []
    for i, cam_name in enumerate(ordering):
        mask_path = os.path.join(source_path, cam_name, 'masks', f'{i:05d}.png')
        hand_mask_path = os.path.join(source_path, cam_name, 'dynamic_masks/0', f'{i:05d}.png')
        mask = cv.imread(mask_path)
        hand_mask = cv.imread(hand_mask_path)
        mask = (mask[:, :, 0] > 0)
        hand_mask = ~(hand_mask[:, :, 0] > 0)
        masks.append((mask & hand_mask).astype(np.float32))
    return masks


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
    elif mask_type == 'full_no_hand':
        masks = get_full_masks_no_hand(source_path, ordering)
    elif mask_type == 'dynamic_no_hand':
        masks = get_dynamic_masks(source_path, ordering, skip_ids=[0])
    else:
        raise ValueError(f'Mask type not supported: {mask_type}')
    return masks
    

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
    return np.mean(psnrs), np.mean(ssims), np.mean(lpipss)


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


def filter_based_on_data_type(settings, data_type):
    filtered_settings = []
    for setting in settings:
        if data_type == 'non_eg' and setting['take_name'] != 'iiith_cooking_111_2':
            filtered_settings.append(setting)
        elif data_type == 'eg' and setting['take_name'] == 'iiith_cooking_111_2':
            filtered_settings.append(setting)
    return filtered_settings


def get_mask_type_based_on_data_type(mask_type, data_type):
    if data_type != 'eg':
        return mask_type
    if mask_type == 'full':
        return 'full_no_hand'
    elif mask_type == 'dynamic':
        return 'dynamic_no_hand'
    return mask_type

if __name__ == '__main__':
    models = ['EgoGaussian', 'Deformable-3D-Gaussians', '4DGaussians', '4d-gaussian-splatting']
    mask_type = 'dynamic'
    data_type = 'eg'  # 'eg' or 'non_eg'

    with open('settings.json', 'r') as f:
        settings = json.load(f)
    settings = filter_based_on_data_type(settings, data_type)
    mask_type = get_mask_type_based_on_data_type(mask_type, data_type)
    print(f'Using mask type: {mask_type}')

    results = {}
    for model in models:
        if model not in results:
            results[model] = {}
        root_path = f'../{model}/output/ego_exo/with_val_set'
        for setting in settings[::-1]:
            scene = setting['take_name']
            if scene not in results[model]:
                results[model][scene] = {}
            for camera_label in ['camera-rgb', 'gopro'] if model != 'EgoGaussian' else ['camera-rgb']:
                if camera_label not in results[model][scene]:
                    results[model][scene][camera_label] = {}
                source_path, gt_path, renders_path = get_paths_from_model(model, root_path, camera_label, scene)
                mean_psnr, mean_ssim, mean_lpips = evaluate(source_path, gt_path, renders_path, mask_type)
                print(f'{model} {camera_label} {scene} PSNR: {mean_psnr} SSIM: {mean_ssim} LPIPS: {mean_lpips}')
                results[model][scene][camera_label] = {'psnr': mean_psnr, 'ssim': mean_ssim, 'lpips': mean_lpips}
    print()
    
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
                mean_over_scenes[model][camera_label][metric] = np.mean(mean_over_scenes[model][camera_label][metric])

    for model in mean_over_scenes:
        for camera_label in mean_over_scenes[model]:
            print(f'{model} {camera_label} PSNR: {mean_over_scenes[model][camera_label]["psnr"]:.2f} SSIM: {mean_over_scenes[model][camera_label]["ssim"]:.2f} LPIPS: {mean_over_scenes[model][camera_label]["lpips"]:.2f}')
    