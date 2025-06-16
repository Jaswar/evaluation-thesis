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

# SSIM implementation based on: https://github.com/VainF/pytorch-msssim/blob/master/pytorch_msssim/ssim.py
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

def get_mask_type_based_on_data_type(mask_type, data_type):
    if data_type != 'eg':
        return mask_type
    if mask_type == 'full':
        return 'full_no_hand'
    elif mask_type == 'dynamic':
        return 'dynamic_no_hand'
    return mask_type