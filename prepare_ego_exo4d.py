import os 
import numpy as np
import projectaria_tools.core.mps as mps
from projectaria_tools.core.mps.utils import filter_points_from_confidence
import pyvista as pv
import cv2 as cv
from projectaria_tools.core import data_provider, calibration
from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions
from projectaria_tools.core.stream_id import StreamId
from plyfile import PlyData, PlyElement
from tqdm import tqdm
import json
import random


def get_vrs_name(path):
    for file in os.listdir(path):
        if file.endswith('.vrs') and 'noimagestreams' not in file:
            return file
    raise ValueError('No .vrs file found')

def get_points(path, num_points):
    global_points_path = os.path.join(path, 'trajectory', 'semidense_points.csv.gz')
    points = mps.read_global_point_cloud(global_points_path)

    # filter the point cloud using thresholds on the inverse depth and distance standard deviation
    inverse_distance_std_threshold = 0.001
    distance_std_threshold = 0.15

    filtered_points = filter_points_from_confidence(points, inverse_distance_std_threshold, distance_std_threshold)

    # example: get position of this point in the world coordinate frame
    points = []
    for point in filtered_points:
        position_world = point.position_world
        points.append(position_world)
    if len(points) > num_points:
        points = random.sample(points, num_points)
    points = np.array(points)
    return points

def undistort_aria(image_array, provider, sensor_name, size, focal_length):
    device_calib = provider.get_device_calibration()
    src_calib = device_calib.get_camera_calib(sensor_name)

    dst_calib = calibration.get_linear_camera_calibration(
        size, size, focal_length
    )

    rectified_array = calibration.distort_by_calibration(
        image_array, dst_calib, src_calib
    )

    return rectified_array

# taken from MonST3R, adapted to no rgb
def storePly(path, xyz):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)
    rgb = np.full((xyz.shape[0], 3), dtype=np.uint8, fill_value=127)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


def get_frames_aria(root_path, provider, camera_label, start_frame, end_frame):
    stream_id = provider.get_stream_id_from_label(camera_label)
    devignetting_mask_folder_path = os.path.join(root_path, 'aria_devignetting_masks')
    device_calib = provider.get_device_calibration()
    device_calib.set_devignetting_mask_folder_path(devignetting_mask_folder_path)
    devignetting_mask = device_calib.load_devignetting_mask(camera_label)

    src_calib = device_calib.get_camera_calib(camera_label)
    focal_length = src_calib.get_focal_lengths()[0]
    size = src_calib.get_image_size()[0]

    frames = []
    for i in range(start_frame, end_frame):
        frame_tuple = provider.get_image_data_by_index(stream_id, i)
        frame = frame_tuple[0].to_numpy_array()

        # load devignetting_mask by camera label
        undistorted_frame = undistort_aria(frame, provider, camera_label, size, focal_length)

        # apply devignetting to source image
        devigneted_image = calibration.devignetting(undistorted_frame, devignetting_mask)

        # rotate the image by 90 degrees
        frame = np.rot90(devigneted_image, k=3)

        frames.append(frame)
    devignetting_mask = np.rot90(devignetting_mask, k=3)
    return frames, devignetting_mask, (focal_length, size / 2)


def read_trajectory(path, provider, camera_label, start_frame, end_frame):
    trajectory = mps.read_closed_loop_trajectory(os.path.join(path, 'trajectory', 'closed_loop_trajectory.csv'))
    stream_id = provider.get_stream_id_from_label(camera_label)
    rgb_stream_label = provider.get_label_from_stream_id(stream_id)
    device_calibration = provider.get_device_calibration()
    rgb_camera_calibration = device_calibration.get_camera_calib(rgb_stream_label)
    rgb_camera_calibration = calibration.rotate_camera_calib_cw90deg(rgb_camera_calibration)
    T_device_rgb_camera = rgb_camera_calibration.get_transform_device_camera()

    timestamps = provider.get_timestamps_ns(stream_id, TimeDomain.DEVICE_TIME)[start_frame:end_frame]
    qvecs = []
    tvecs = []
    for query_timestamp in timestamps:
        pose_info = mps.utils.get_nearest_pose(trajectory, query_timestamp)
        T_world_device = pose_info.transform_world_device
        T_world_rgb_camera = T_world_device @ T_device_rgb_camera
        [qw, qx, qy, qz, tx, ty, tz] = T_world_rgb_camera.to_quat_and_translation()[0]
        qvecs.append([qw, qx, qy, qz])
        tvecs.append([tx, ty, tz])
    qvecs = np.array(qvecs)
    tvecs = np.array(tvecs)
    return qvecs, tvecs

def prepare_aria(root_path, seq_name, out_path, camera_label, start_frame, end_frame, target_size=384, num_points=10000):
    os.makedirs(os.path.join(out_path, camera_label, seq_name), exist_ok=True)
    os.makedirs(os.path.join(out_path, camera_label, seq_name, camera_label, 'frames'), exist_ok=True)
    os.makedirs(os.path.join(out_path, camera_label, seq_name, camera_label, 'masks'), exist_ok=True)

    seq_path = os.path.join(root_path, 'takes', seq_name)
    vrs_name = get_vrs_name(seq_path)
    provider = data_provider.create_vrs_data_provider(os.path.join(seq_path, vrs_name))

    points = get_points(seq_path, num_points)
    frames, devignetting_mask, intrs = get_frames_aria(root_path, provider, camera_label, start_frame, end_frame)
    qvecs, tvecs = read_trajectory(seq_path, provider, camera_label, start_frame, end_frame)
    assert len(frames) == len(qvecs) == len(tvecs) == end_frame - start_frame

    devignetting_mask = np.stack([devignetting_mask, devignetting_mask, devignetting_mask], axis=2)
    devignetting_mask = cv.threshold(devignetting_mask, 0, 255, cv.THRESH_BINARY)[1]
    devignetting_mask = devignetting_mask.astype(np.uint8)

    # resize frames
    devignetting_mask = cv.resize(devignetting_mask, (target_size, target_size))
    factor = target_size / frames[0].shape[0]
    for i, frame in enumerate(frames):
        frames[i] = cv.resize(frame, (target_size, target_size))
    intrs = (intrs[0] * factor, intrs[1] * factor)
    
    for i in range(len(frames)):
        cv.imwrite(os.path.join(out_path, camera_label, seq_name, camera_label, 'masks', f'{i:05d}.png'), devignetting_mask)
    for i, frame in enumerate(frames):
        cv.imwrite(os.path.join(out_path, camera_label, seq_name, camera_label, 'frames', f'{i:05d}.png'), cv.cvtColor(frame, cv.COLOR_RGB2BGR))

    with open(os.path.join(out_path, camera_label, seq_name, camera_label, 'trajectory.txt'), 'w') as f:
        f.write(f'QW QX QY QZ TX TY TZ\n')
        for i in range(len(qvecs)):
            f.write(f'{qvecs[i][0]} {qvecs[i][1]} {qvecs[i][2]} {qvecs[i][3]} {tvecs[i][0]} {tvecs[i][1]} {tvecs[i][2]}\n')
    with open(os.path.join(out_path, camera_label, seq_name, camera_label, 'intrinsics.txt'), 'w') as f:
        f.write(f'FX FY CX CY\n')
        for _ in range(len(frames)):
            f.write(f'{intrs[0]} {intrs[0]} {intrs[1]} {intrs[1]}\n')

    storePly(os.path.join(out_path, camera_label, seq_name, 'points.ply'), points)

    with open(os.path.join(out_path, camera_label, seq_name, 'ordering.txt'), 'w') as f:
        for i in range(len(frames)):
            f.write(f'{camera_label}\n')

def undistort_exocam(image, intrinsics, distortion_coeffs, dimension=(3840, 2160)):
    DIM = dimension
    dim2 = None
    dim3 = None
    balance = 0.8
    # Load the distortion parameters
    distortion_coeffs = distortion_coeffs
    # Load the camera intrinsic parameters
    intrinsics = intrinsics

    dim1 = image.shape[:2][::-1]  # dim1 is the dimension of input image to un-distort

    # Change the calibration dim dynamically (bouldering cam01 and cam04 are verticall for examples)
    if DIM[0] != dim1[0]:
        DIM = (DIM[1], DIM[0])

    assert (
        dim1[0] / dim1[1] == DIM[0] / DIM[1]
    ), "Image to undistort needs to have same aspect ratio as the ones used in calibration"
    if not dim2:
        dim2 = dim1
    if not dim3:
        dim3 = dim1
    scaled_K = (
        intrinsics * dim1[0] / DIM[0]
    )  # The values of K is to scale with image dimension.
    scaled_K[2][2] = 1.0  # Except that K[2][2] is always 1.0

    # This is how scaled_K, dim2 and balance are used to determine the final K used to un-distort image. OpenCV document failed to make this clear!
    new_K = cv.fisheye.estimateNewCameraMatrixForUndistortRectify(
        scaled_K, distortion_coeffs, dim2, np.eye(3), balance=balance
    )
    map1, map2 = cv.fisheye.initUndistortRectifyMap(
        scaled_K, distortion_coeffs, np.eye(3), new_K, dim3, cv.CV_16SC2
    )
    undistorted_image = cv.remap(
        image,
        map1,
        map2,
        interpolation=cv.INTER_LINEAR,
        borderMode=cv.BORDER_CONSTANT,
    )

    return undistorted_image, new_K


def get_frames_gopro(seq_path, camera_label, start_frame, end_frame):
    video_path = os.path.join(seq_path, 'frame_aligned_videos', camera_label + '.mp4')
    cap = cv.VideoCapture(video_path)
    frames = []
    for i in range(0, end_frame):
        ret, frame = cap.read()
        if not ret:
            print(f'Error reading frame {i}')
            break
        if i < start_frame:
            continue
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    return frames


def get_go_pro_calib(seq_path, camera_label):
    with open(os.path.join(seq_path, 'trajectory', 'gopro_calibs.csv'), 'r') as f:
        lines = f.read().split('\n')
    lines = [line.split(',') for line in lines if line != '']
    all_calibs = {}
    for line in lines[1:]:
        if line[0] not in all_calibs:
            all_calibs[line[0]] = {}
        for i, param in enumerate(line):
            all_calibs[line[0]][lines[0][i]] = param

    qvecs = []
    tvecs = []
    intrinsics = []
    distortion_coeffs = []
    names = []
    for key, cam_calib in all_calibs.items():
        qvec = [float(cam_calib['qw_world_cam']), 
                float(cam_calib['qx_world_cam']), 
                float(cam_calib['qy_world_cam']), 
                float(cam_calib['qz_world_cam'])]
        tvec = [float(cam_calib['tx_world_cam']),
                float(cam_calib['ty_world_cam']),
                float(cam_calib['tz_world_cam'])]
        intrs = np.array(
            [
                [float(cam_calib["intrinsics_0"]), 0, float(cam_calib["intrinsics_2"])],
                [0, float(cam_calib["intrinsics_1"]), float(cam_calib["intrinsics_3"])],
                [0, 0, 1],
            ]
        )
        distrs = np.array(
            [
                float(cam_calib["intrinsics_4"]),
                float(cam_calib["intrinsics_5"]),
                float(cam_calib["intrinsics_6"]),
                float(cam_calib["intrinsics_7"]),
            ]
        )
        names.append(key)
        qvecs.append(qvec)
        tvecs.append(tvec)
        intrinsics.append(intrs)
        distortion_coeffs.append(distrs)
    return names, qvecs, tvecs, intrinsics, distortion_coeffs


def prepare_gopro(root_path, seq_name, out_path, camera_label, start_frame, end_frame, target_height=384, num_points=10000):
    camera_label = 'gopro'
    os.makedirs(os.path.join(out_path, camera_label, seq_name), exist_ok=True)

    seq_path = os.path.join(root_path, 'takes', seq_name)
    names, qvecs, tvecs, intrs, distortions = get_go_pro_calib(seq_path, camera_label)
    points = get_points(seq_path, num_points)

    for (name, qvec, tvec, intr, dists) in zip(names, qvecs, tvecs, intrs, distortions):
        os.makedirs(os.path.join(out_path, camera_label, seq_name, name), exist_ok=True)
        os.makedirs(os.path.join(out_path, camera_label, seq_name, name, 'frames'), exist_ok=True)
        os.makedirs(os.path.join(out_path, camera_label, seq_name, name, 'masks'), exist_ok=True)
        frames = get_frames_gopro(seq_path, name, start_frame, end_frame)

        devignetting_mask = np.ones((frames[0].shape[0], frames[0].shape[1], 3), dtype=np.uint8) * 255
        devignetting_mask, new_intrs = undistort_exocam(devignetting_mask, intr, dists, (frames[0].shape[1], frames[0].shape[0]))

        undistored_frames = []
        for i, frame in tqdm(enumerate(frames)):
            undistored_frame, _ = undistort_exocam(frame, intr, dists, (frames[0].shape[1], frames[0].shape[0]))
            undistored_frames.append(undistored_frame)
        
        new_intrs = [new_intrs[0, 0], new_intrs[1, 1], new_intrs[0, 2], new_intrs[1, 2]]

        target_width = int(target_height * frames[0].shape[1] / frames[0].shape[0])
        new_intrs = (new_intrs[0] * target_width / frames[0].shape[1], new_intrs[1] * target_height / frames[0].shape[0],
                new_intrs[2] * target_width / frames[0].shape[1], new_intrs[3] * target_height / frames[0].shape[0])
        devignetting_mask = cv.resize(devignetting_mask, (target_width, target_height))
        for i, frame in enumerate(undistored_frames):
            undistored_frames[i] = cv.resize(frame, (target_width, target_height))

        for i, frame in enumerate(undistored_frames):
            cv.imwrite(os.path.join(out_path, camera_label, seq_name, name, 'masks', f'{i:05d}.png'), devignetting_mask)
            cv.imwrite(os.path.join(out_path, camera_label, seq_name, name, 'frames', f'{i:05d}.png'), cv.cvtColor(frame, cv.COLOR_RGB2BGR))

        with open(os.path.join(out_path, camera_label, seq_name, name, 'trajectory.txt'), 'w') as f:
            f.write(f'QW QX QY QZ TX TY TZ\n')
            for _ in range(len(frames)):
                f.write(f'{qvec[0]} {qvec[1]} {qvec[2]} {qvec[3]} {tvec[0]} {tvec[1]} {tvec[2]}\n')
        with open(os.path.join(out_path, camera_label, seq_name, name, 'intrinsics.txt'), 'w') as f:
            f.write(f'FX FY CX CY\n')
            for _ in range(len(frames)):
                f.write(f'{new_intrs[0]} {new_intrs[1]} {new_intrs[2]} {new_intrs[3]}\n')

    storePly(os.path.join(out_path, camera_label, seq_name, 'points.ply'), points)

    # ordering = [(i % (num_cams * 2)) // 2 for i in range(len(all_frames[0]))]
    ordering = np.random.randint(1, len(names), end_frame - start_frame)
    for i in range(len(ordering)):
        if i % 2 == 1:
            ordering[i] = 0
    with open(os.path.join(out_path, camera_label, seq_name, 'ordering.txt'), 'w') as f:
        for i in ordering:
            f.write(f'{names[i]}\n')


def main(root_path, seq_name, out_path, camera_label, start_time, end_time, target_size=384):
    seq_path = os.path.join(root_path, 'takes', seq_name)
    vrs_name = get_vrs_name(seq_path)
    provider = data_provider.create_vrs_data_provider(os.path.join(seq_path, vrs_name))

    stream_id = provider.get_stream_id_from_label('camera-rgb')
    timestamps = provider.get_timestamps_ns(stream_id, TimeDomain.DEVICE_TIME)
    timestamps = np.array([ts / 1e9 for ts in timestamps]) - timestamps[0] / 1e9
    start_frame = np.argmax(timestamps > start_time)
    end_frame = np.argmax(timestamps > end_time) # upper index will be excluded
    print(f'Processing sequence {seq_name} from frame {start_frame} (included) to {end_frame} (excluded)')
    
    if camera_label == 'camera-rgb':
        prepare_aria(root_path, seq_name, out_path, camera_label, start_frame, end_frame, target_size)
    elif camera_label == 'gopro':
        prepare_gopro(root_path, seq_name, out_path, camera_label, start_frame, end_frame, target_size)
    else:
        raise ValueError('Unknown camera label')

if __name__ == '__main__':
    root_path = 'ego_exo_4d'
    json_file_name = 'settings.json'
    out_path = 'output/tmp'
    with open(json_file_name, 'r') as f:
        settings = json.load(f)
    for seq in settings:
        np.random.seed(42)
        random.seed(42)
        seq_name = seq['take_name']
        if seq_name != 'iiith_cooking_58_2':
            continue
        start_time = seq['start_time']
        end_time = seq['end_time']
        camera_label = 'camera-rgb'  # gopro for GoPro, camera-rgb for Aria
        main(root_path, seq_name, out_path, camera_label, start_time, end_time)
        camera_label = 'gopro'  # gopro for GoPro, camera-rgb for Aria
        main(root_path, seq_name, out_path, camera_label, start_time, end_time)
        