import os
import json
import numpy as np

from common import get_masks, psnr, ssim, lpips, get_paths_from_model, get_mask_type_based_on_data_type

import matplotlib.pyplot as plt
import cv2 as cv
import scipy

# from https://mariogc.com/post/angular-velocity-quaternions/#example
def angular_velocity(q1, q2, dt=1/30.):
    return (2 / dt) * np.array([
        q1[0]*q2[1] - q1[1]*q2[0] - q1[2]*q2[3] + q1[3]*q2[2],
        q1[0]*q2[2] + q1[1]*q2[3] - q1[2]*q2[0] - q1[3]*q2[1],
        q1[0]*q2[3] - q1[1]*q2[2] + q1[2]*q2[1] - q1[3]*q2[0]])

def linear_velocity(t1, t2, dt=1/30.):
    return (1 / dt) * np.array([
        t2[0] - t1[0],
        t2[1] - t1[1],
        t2[2] - t1[2]
    ])

def get_velocities(xs, linear):
    vels = []
    for i in range(len(xs) - 1):
        if not linear:
            vel = angular_velocity(xs[i], xs[i + 1])
        else:
            vel = linear_velocity(xs[i], xs[i + 1])
        vels.append(vel)
    return np.array(vels)


def get_extr(source_path):
    with open(os.path.join(source_path, 'ordering.txt'), 'r') as f:
        ordering = f.read().split('\n')
    ordering = [o for o in ordering if o != '']
    
    ts, qs = [], []
    for i, cam_name in enumerate(ordering):
        with open(os.path.join(source_path, cam_name, 'trajectory.txt')) as file:
            traj = file.read().split('\n')
        traj = [t for t in traj if t != ''][1:]
        extr = list(map(float, traj[i].split(' ')))
        qs.append(extr[:4])
        ts.append(extr[4:])
    return qs, ts

def plot_velocities(rv, tv):
    plt.plot(rv[:, 0], label='$\\omega_x$')
    plt.plot(rv[:, 1], label='$\\omega_y$')
    plt.plot(rv[:, 2], label='$\\omega_z$')
    plt.legend()
    plt.show()

    plt.plot(tv[:, 0], label='$v_x$')
    plt.plot(tv[:, 1], label='$v_y$')
    plt.plot(tv[:, 2], label='$v_z$')
    plt.legend()
    plt.show()

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

    with open('settings.json', 'r') as f:
        settings = json.load(f)

    
    for model in models:
        rvs = []
        tvs = []
        all_psnrs = []
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

                qs, ts = get_extr(source_path)  
                rv = get_velocities(qs, linear=False)[2::4]
                rv = np.abs(rv)
                rv = normalize(rv)
                tv = get_velocities(ts, linear=True)[2::4]
                tv = np.abs(tv)
                tv = normalize(tv)
                # plot_velocities(rv, tv)

                rv = np.max(rv, axis=1)
                tv = np.max(tv, axis=1)
                psnrs, ssims, lpipss = evaluate(source_path, gt_path, renders_path, task_specific_mask)
                psnrs = normalize(lpipss, make_log=False)
                # rv = rv[2::4]
                # tv = tv[2::4]
                # plt.plot(tv, label='v')
                # plt.plot(psnrs, label='PSNR')
                # plt.legend()
                # plt.show()
                #plt.scatter(tv, psnrs, label=scene)
                rvs.extend(rv)
                tvs.extend(tv)
                all_psnrs.extend(psnrs)
        rvs = np.array(rvs)
        tvs = np.array(tvs)
        all_psnrs = np.array(all_psnrs)
        plt.scatter(tvs, all_psnrs, label=model)
    plt.plot([np.min(tvs) - 0.03, 1.0], [1.0, 0.0], 'k--')
    plt.legend()
    plt.xlabel('Log Translation Velocity')
    plt.ylabel('Log PSNR')
    plt.show()

    # plt.hist(tvs)
    # plt.show()
    # plt.hist(all_psnrs)
    # plt.show()

    print(scipy.stats.pearsonr(tvs, all_psnrs))

    
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')

    # ax.scatter(tvs, rvs, all_psnrs, s=20)

    # ax.set_xlabel('TVS')
    # ax.set_ylabel('RVS')
    # ax.set_zlabel('PSNR')

    # plt.show()

if __name__ == '__main__':
    main()