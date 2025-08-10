import os 
from colmap_loader import write_extrinsics_binary, qvec2rotmat, focal2fov, rotmat2qvec, Camera, write_intrinsics_binary, Image
import numpy as np
import cv2 as cv
import json


def convIntrinsicsAndExtrinsics(ego_exo_path, output_path):
    with open(os.path.join(ego_exo_path, 'ordering.txt'), 'r') as f:
        ordering = f.read().split('\n')
    ordering = [line for line in ordering if line != '']
    cam_names = list(set(ordering))

    all_intrinsics = {}
    all_extrinsics = {}
    for i, cam_name in enumerate(cam_names):
        with open(os.path.join(ego_exo_path, cam_name, 'intrinsics.txt'), 'r') as f:
            intrinsics = f.read().split('\n')
        intrinsics = [line for line in intrinsics if line != ''][1:]
        intrinsics = [list(map(float, line.split(' '))) for line in intrinsics]
        all_intrinsics[cam_name] = intrinsics[0]

        with open(os.path.join(ego_exo_path, cam_name, 'trajectory.txt'), 'r') as f:
            extrinsics = f.read().split('\n')
        extrinsics = [line for line in extrinsics if line != ''][1:]
        extrinsics = [list(map(float, line.split(' '))) for line in extrinsics]
        all_extrinsics[cam_name] = extrinsics
    reference_frame = cv.imread(os.path.join(ego_exo_path, ordering[0], 'frames', '00000.png'))
    height, width = reference_frame.shape[:2]

    # save the instrinsics
    colmap_cams = {}
    id_name_mapping = {}
    for i, cam_name in enumerate(all_intrinsics):
        fx = all_intrinsics[cam_name][0]
        fy = all_intrinsics[cam_name][1]
        cx = all_intrinsics[cam_name][2]
        cy = all_intrinsics[cam_name][3]
        cam = Camera(id=i + 1, model='PINHOLE', width=width, height=height, params=[fx, fy, cx, cy])
        colmap_cams[i + 1] = cam
        id_name_mapping[cam_name] = i + 1
    write_intrinsics_binary(colmap_cams, os.path.join(output_path, 'sparse/0/cameras.bin'))

    colmap_images = {}
    for i, order in enumerate(ordering):
        qvec = all_extrinsics[order][i][:4]
        tvec = all_extrinsics[order][i][4:]

        R = qvec2rotmat(qvec)
        T = np.array(tvec)
        c2w = np.zeros((4, 4))
        c2w[:3, :3] = R
        c2w[:3, 3] = T
        c2w[3, 3] = 1
        w2c = np.linalg.inv(c2w)
        R = w2c[:3,:3]
        T = w2c[:3, 3]
        qvec = rotmat2qvec(R)
        tvec = T

        filename = f'{i:05d}.png'
        image = Image(id=i+1, qvec=qvec, tvec=tvec, camera_id=id_name_mapping[order], name=filename, xys=[], point3D_ids=[])
        colmap_images[i+1] = image
    write_extrinsics_binary(colmap_images, os.path.join(output_path, 'sparse/0/images.bin'))


def movePly(ego_exo_path, output_path):
    os.system(f'cp \"{ego_exo_path}/points.ply\" \"{output_path}/sparse/0/points3D.ply\"')


def moveFrames(ego_exo_path, output_path):
    source_path = os.path.join(ego_exo_path, 'camera-rgb/frames')
    target_path = os.path.join(output_path, 'images')
    os.system(f'cp {source_path}/* {target_path}')

def moveMasks(ego_exo_path, output_path):
    person_masks = sorted(os.listdir(os.path.join(ego_exo_path, 'camera-rgb/dynamic_masks/0')))
    camera_masks = sorted(os.listdir(os.path.join(ego_exo_path, 'camera-rgb/masks')))
    for i, (person_mask_path, camera_mask_path) in enumerate(zip(person_masks, camera_masks)):
        person_mask = cv.imread(os.path.join(ego_exo_path, 'camera-rgb/dynamic_masks/0', person_mask_path))
        camera_mask = cv.imread(os.path.join(ego_exo_path, 'camera-rgb/masks', camera_mask_path))
        hand_mask = cv.bitwise_or(person_mask, ~camera_mask)
        hand_mask_path = os.path.join(output_path, 'hand_masks', f'{i:05d}.png')
        cv.imwrite(hand_mask_path, hand_mask)
    obj_masks = sorted(os.listdir(os.path.join(ego_exo_path, 'camera-rgb/dynamic_masks/1')))
    for obj_mask_path in obj_masks:
        os.system(f'cp \"{os.path.join(ego_exo_path, "camera-rgb/dynamic_masks/1", obj_mask_path)}\" \"{os.path.join(output_path, "obj_masks")}\"')


def createSplits(ego_exo_path, output_path):
    with open(os.path.join(ego_exo_path, 'dynamics.json'), 'r') as f:
        split = json.load(f)
    dynamic_range = None
    for obj in split:
        if obj['id'] == 1:
            dynamic_range = obj['ranges']
    assert len(dynamic_range) == 1
    dynamic_range = dynamic_range[0]
    num_frames = len(os.listdir(os.path.join(output_path, 'images')))
    
    with open(os.path.join(output_path, 'split/phase_frame_index.txt'), 'w') as f:
        f.write(f'0,{dynamic_range[0] - 1}\n')
        f.write(f'{dynamic_range[0]},{dynamic_range[1]}\n')
        f.write(f'{dynamic_range[1] + 1},{num_frames - 1}\n')

    training_indices = list(range(num_frames))[::2]
    testing_indices = list(range(num_frames))[3::4]
    assert set(training_indices).isdisjoint(testing_indices)

    static_eval_indices = []
    dynamic_eval_indices = []
    for inx in testing_indices:
        if dynamic_range[0] <= inx <= dynamic_range[1]:
            dynamic_eval_indices.append(inx)
        else:
            static_eval_indices.append(inx)
    
    with open(os.path.join(output_path, 'split/training_frames.txt'), 'w') as f:
        f.write('\n'.join(map(str, training_indices)))
    with open(os.path.join(output_path, 'split/dynamic_eval_frames.txt'), 'w') as f:
        f.write('\n'.join(map(str, dynamic_eval_indices)))
    with open(os.path.join(output_path, 'split/static_eval_frames.txt'), 'w') as f:
        f.write('\n'.join(map(str, static_eval_indices)))

    with open(os.path.join(output_path, 'frames.txt'), 'w') as f:
        f.write(f'0-{num_frames - 1}')


def conv2EgoGaussian(ego_exo_path, output_path):
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(os.path.join(output_path, 'sparse/0'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'hand_masks'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'obj_masks'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'split'), exist_ok=True)

    convIntrinsicsAndExtrinsics(ego_exo_path, output_path)
    movePly(ego_exo_path, output_path)
    moveFrames(ego_exo_path, output_path)
    moveMasks(ego_exo_path, output_path)
    createSplits(ego_exo_path, output_path)

if __name__ == '__main__':
    ego_exo_path = 'output/all_saves/camera-rgb/uniandes_cooking_005_4'
    output_path = 'output/HOI4D/uniandes_cooking_005_4'
    conv2EgoGaussian(ego_exo_path, output_path)




