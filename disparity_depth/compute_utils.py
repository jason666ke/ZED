import time

import cv2
import numpy as np
import open3d as o3d


def compute_disparity(left, right,
                      min_disp, num_disp,
                      block_size,
                      P1, P2,
                      disp12MaxDiff,
                      preFilterCap,
                      uniquenessRatio,
                      speckleWindowSize, speckleRange,
                      mode):
    # 创建一个StereoSGBM实例
    stereo = cv2.StereoSGBM.create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=block_size,
        P1=P1,
        P2=P2,
        disp12MaxDiff=disp12MaxDiff,
        preFilterCap=preFilterCap,
        uniquenessRatio=uniquenessRatio,
        speckleWindowSize=speckleWindowSize,
        speckleRange=speckleRange,
        mode=mode
    )

    # 将整数的视差值转化为浮点数，便于更好的表示
    # disp = stereo.compute(left, right).astype(np.float32) / 16.0
    disp = stereo.compute(left, right).astype(np.float32)
    # 视差图归一化到 [0, 1]的范围内
    # disp = np.clip((disp - min_disp) / num_disp, 0, 1)

    return disp


# 深度图
def compute_depth(disparity_map, baseline, fx):
    """
    Compute depth map from disparity map
    :param disparity_map:
    :param baseline: baseline of camera (unit: mm)
    :param fx: focal length of camera
    :return: depth map
    """
    # Depth = 焦距 * 基线距离 / 视差
    depth_map = (fx * baseline) / disparity_map
    # depth_map = np.zeros_like(disparity_map)
    # for row in range(disparity_map.shape[0]):
    #     for col in range(disparity_map.shape[1]):
    #         if disparity_map[row, col] != 0:
    #             depth_map[row, col] = (fx * baseline) / disparity_map[row, col]

    return depth_map


def depth2pcd(depth_map, camera_intrinsics):
    # todo: 加入上色代码
    width = camera_intrinsics.width
    height = camera_intrinsics.height
    intrinsic_matrix = camera_intrinsics.intrinsic_matrix
    fx = intrinsic_matrix[0][0]
    fy = intrinsic_matrix[1][1]
    cx = intrinsic_matrix[0][2]
    cy = intrinsic_matrix[1][2]
    # width, height, fx, fy, cx, cy = camera_intrinsics
    # Generate pixel grid
    u, v = np.meshgrid(np.arange(width), np.arange(height))

    # compute x, y, z coordinates
    x = (u - cx) * depth_map / fx
    y = (v - cy) * depth_map / fy
    z = depth_map

    pcd = np.dstack((x, y, z))

    return pcd


def depth2pcd_with_o3d(depth_map, intrinsic):
    depth_image = o3d.geometry.Image(depth_map)
    # print(depth_image.type, intrinsic.type)
    pcd = o3d.geometry.PointCloud.create_from_depth_image(depth=depth_image, intrinsic=intrinsic)
    return pcd


def visualize_pcd(point_cloud):
    # pcd = o3d.io.read_point_cloud(point_cloud)
    o3d.visualization.draw([point_cloud])
    # o3d.visualization.draw_geometries([point_cloud])


def length_to_pixels(length_mm, sensor_width_mm, image_width):
    """
    Convert length from millimeters to pixels.

    Parameters:
    :param length_mm: Length in millimeters.
    :param sensor_width_mm: Sensor width in millimeters.
    :param image_width: Image width in pixels

    Returns:
    :return pixels: Length in pixels
    """
    pixels = length_mm / (sensor_width_mm / image_width)
    return pixels


def write_ply(point_cloud, save_ply):
    start = time.time()
    float_formatter = lambda x: "%.4f" % x
    points = []

    for i in point_cloud.T:
        points.append("{} {} {} {} {} {} o\n".format(
            float_formatter(i[0]), float_formatter(i[1]), float_formatter(i[2]),
            int(i[3]), int(i[4]), int(i[5])
        ))

    file = open(save_ply, "w")
    file.write(
        '''ply
        format ascii 1.0
        element vertex %d
        property float x
        property float y
        property float z
        property uchar red
        property uchar green
        property uchar blue
        property uchar alpha
        end_header
        %s
        ''' % (len(points), "".join(points))
    )
    file.close()

    end = time.time()
    print("Write into .ply file Done", end - start)