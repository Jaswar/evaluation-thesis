import os
import json
import numpy as np

from common import get_masks, psnr, ssim, lpips, get_paths_from_model, get_mask_type_based_on_data_type

import matplotlib
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
    linear = False  # True for translational velocity, False for rotational/angular
    models = ['Deformable-3D-Gaussians', '4DGaussians', '4d-gaussian-splatting', 'EgoGaussian']
    model_mapping = {
        'EgoGaussian': 'EgoGaussian',
        'Deformable-3D-Gaussians': 'Def3DGS',
        '4DGaussians': '4DGS',
        '4d-gaussian-splatting': 'RTGS',
    }

    with open('settings.json', 'r') as f:
        settings = json.load(f)
    
    bin_size = 0.05
    plt.rcParams.update({'font.size': 22})
    scatter_csvs = {}
    for model in models:
        scatter_csvs[model] = 'velocity,lpips\n'
    trend_csvs = {}
    for model in models:
        trend_csvs[model] = 'velocity,lpips\n'
    for model in models:
        all_velocities = []
        all_lpipss = []
        bins = {}
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
                rv = get_velocities(qs, linear=False)[2::4]  # start at 2, because v_2 represents the velocity between frames 2 and 3
                rv = np.abs(rv)
                rv = normalize(rv)
                tv = get_velocities(ts, linear=True)[2::4]
                tv = np.abs(tv)
                tv = normalize(tv)
                # plot_velocities(rv, tv)

                rv = np.max(rv, axis=1)
                tv = np.max(tv, axis=1)
                psnrs, ssims, lpipss = evaluate(source_path, gt_path, renders_path, task_specific_mask)
                lpipss = normalize(lpipss, make_log=False)
                # rv = rv[2::4]
                # tv = tv[2::4]
                # plt.plot(tv, label='v')
                # plt.plot(psnrs, label='PSNR')
                # plt.legend()
                # plt.show()
                #plt.scatter(tv, psnrs, label=scene)
                if linear:
                    all_velocities.extend(tv)
                else:
                    all_velocities.extend(rv)
                all_lpipss.extend(lpipss)
        for v, l in zip(all_velocities, all_lpipss):
            # do the binning/histogram
            b = v // bin_size + 1
            if b not in bins:
                bins[b] = []
            bins[b].append(l)

            # add each entry to the .csv file
            scatter_csvs[model] += f'{v},{l}\n'

        all_velocities = np.array(all_velocities)
        all_lpipss = np.array(all_lpipss)
        # np.random.shuffle(all_lpipss)
        print(f'Spearman statistics: {scipy.stats.spearmanr(all_velocities, all_lpipss)}')
        print(f'Pearson statistics for translation: {scipy.stats.pearsonr(all_velocities, all_lpipss)}')
        plt.scatter(all_velocities, all_lpipss, label=model_mapping[model], s=10)
        bins = list(sorted(bins.items(), key=lambda x: x[0]))
        bins = [(b[0], np.mean(b[1])) for b in bins]
        xs = [b[0] * bin_size for b in bins]
        ys = [b[1] for b in bins]
        plt.plot(xs, ys, label=model_mapping[model], linewidth=5, linestyle='dashed')
        for v, l in zip(xs, ys):
            trend_csvs[model] += f'{v},{l}\n'
    plt.legend()
    if linear:
        plt.xlabel('Log Linear Velocity')
    else:
        plt.xlabel('Log Angular Velocity')
    plt.ylabel('LPIPS')
    plt.show()

    for model in models:
        filename = f'linear_scatter_data_{model_mapping[model]}.csv' if linear else f'angular_scatter_data_{model_mapping[model]}.csv'
        with open(os.path.join('scatter_csvs', filename), 'w') as f:
            f.write(scatter_csvs[model])
        filename = f'linear_trend_data_{model_mapping[model]}.csv' if linear else f'angular_trend_data_{model_mapping[model]}.csv'
        with open(os.path.join('trend_csvs', filename), 'w') as f:
            f.write(trend_csvs[model])

if __name__ == '__main__':
    main()