import os
import json
import numpy as np

from common import get_masks, psnr, ssim, lpips, get_paths_from_model, get_mask_type_based_on_data_type

import matplotlib
import matplotlib.pyplot as plt
import cv2 as cv
import scipy
from sklearn.linear_model import LinearRegression

def to_latex_scientific(val):
    if isinstance(val, (int, float)):
        s = f"{val:.1e}"
        base, exp = s.split('e')
        exp = int(exp)
        return f"${base} \\cdot 10^{{{exp}}}$"
    return str(val)

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


def get_linear_baseline(ts):
    max_dist = 0
    for t1 in ts:
        for t2 in ts:
            dist = (t1[0] - t2[0]) ** 2 + (t1[1] - t2[1]) ** 2 + (t1[2] - t2[2]) ** 2
            max_dist = max(max_dist, dist)
    return max_dist

def quaternion_diff(q1, q2):
    q1 = np.array(q1)
    q2 = np.array(q2)
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)
    dot_product = np.abs(np.dot(q1, q2))
    dot_product = np.clip(dot_product, -1.0, 1.0)  # for numerical stability
    return 2 * np.arccos(dot_product)

def get_angular_baseline(qs):
    max_dist = 0
    for q1 in qs:
        for q2 in qs:
            angle = quaternion_diff(q1, q2)
            max_dist = max(max_dist, angle)
    return max_dist


def normalize(arr, make_log=True):
    if make_log:
        arr = np.log(arr)
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))

def main():
    linear = True  # True for translational velocity, False for rotational/angular
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
    baseline_csvs = {}
    for model in models:
        baseline_csvs[model] = 'baseline,lpips\n'
    baseline_linear_csvs = {}
    for model in models:
        baseline_linear_csvs[model] = 'baseline,lpips\n'
    for model in models:
        all_velocities = []
        all_lpipss = []
        bins = {}
        baselines = []
        mean_lpipss = []
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
                if linear:
                    baseline = get_linear_baseline(ts)
                else:
                    baseline = get_angular_baseline(qs)
                baselines.append(baseline)
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
                lpipss = psnrs
                mean_lpipss.append(np.mean(lpipss))
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
        baselines = np.array(baselines)
        mean_lpipss = np.array(mean_lpipss)
        # baselines = np.log(baselines)
        # np.random.shuffle(all_lpipss)
        # np.random.shuffle(mean_lpipss)

        X = baselines.reshape(-1, 1)
        y = mean_lpipss.reshape(-1, 1)
        reg = LinearRegression().fit(X, y)
        plt.scatter(baselines, mean_lpipss, label=model_mapping[model])
        endpoints = np.array([[np.min(baselines)], [np.max(baselines)]])
        preds = reg.predict(endpoints)
        plt.plot(endpoints.reshape(-1), preds.reshape(-1), linestyle='dashed', label=model_mapping[model])
        print(f'Linear regression output: {endpoints=}, {preds=}')
        baseline_linear_csvs[model] += f'{endpoints[0][0]},{preds[0][0]}\n'
        baseline_linear_csvs[model] += f'{endpoints[1][0]},{preds[1][0]}\n'

        pearson_vel = scipy.stats.pearsonr(all_velocities, all_lpipss)
        spearman_vel = scipy.stats.spearmanr(all_velocities, all_lpipss)
        print(f'Pearson statistics for velocity: ${round(float(pearson_vel.statistic), 2)}$, {to_latex_scientific(float(pearson_vel.pvalue))}')
        print(f'Spearman statistics for velocity: ${round(float(spearman_vel.statistic), 2)}$, {to_latex_scientific(float(spearman_vel.pvalue))}')

        pearson_baseline = scipy.stats.pearsonr(baselines, mean_lpipss)
        spearman_baseline = scipy.stats.spearmanr(baselines, mean_lpipss)
        print(f'Pearson statistics for baseline: ${round(float(pearson_baseline.statistic), 2)}$, {to_latex_scientific(float(pearson_baseline.pvalue))}')
        print(f'Spearman statistics for baseline: ${round(float(spearman_baseline.statistic), 2)}$, {to_latex_scientific(float(spearman_baseline.pvalue))}')
        # plt.scatter(all_velocities, all_lpipss, label=model_mapping[model], s=10)
        bins = list(sorted(bins.items(), key=lambda x: x[0]))
        bins = [(b[0], np.mean(b[1])) for b in bins]
        xs = [b[0] * bin_size for b in bins]
        ys = [b[1] for b in bins]
        # plt.plot(xs, ys, label=model_mapping[model], linewidth=5, linestyle='dashed')
        for v, l in zip(xs, ys):
            trend_csvs[model] += f'{v},{l}\n'
        for b, l in zip(baselines, mean_lpipss):
            baseline_csvs[model] += f'{b},{l}\n'
    # plt.legend()
    # if linear:
    #     plt.xlabel('Log Linear Velocity')
    # else:
    #     plt.xlabel('Log Angular Velocity')
    # plt.ylabel('LPIPS')
    # plt.show()

    plt.legend()
    plt.show()

    for model in models:
        # velocity scatter plot and trend line
        filename = f'linear_scatter_data_{model_mapping[model]}.csv' if linear else f'angular_scatter_data_{model_mapping[model]}.csv'
        with open(os.path.join('scatter_csvs', filename), 'w') as f:
            f.write(scatter_csvs[model])
        filename = f'linear_trend_data_{model_mapping[model]}.csv' if linear else f'angular_trend_data_{model_mapping[model]}.csv'
        with open(os.path.join('trend_csvs', filename), 'w') as f:
            f.write(trend_csvs[model])
        # camera baseline data
        filename = f'linear_baseline_data_{model_mapping[model]}.csv' if linear else f'angular_baseline_data_{model_mapping[model]}.csv'
        with open(os.path.join('baseline_csvs', filename), 'w') as f:
            f.write(baseline_csvs[model])
        # camera baseline linear model
        filename = f'linear_baseline_linear_data_{model_mapping[model]}.csv' if linear else f'angular_baseline_linear_data_{model_mapping[model]}.csv'
        with open(os.path.join('baseline_linear_csvs', filename), 'w') as f:
            f.write(baseline_linear_csvs[model])

if __name__ == '__main__':
    main()