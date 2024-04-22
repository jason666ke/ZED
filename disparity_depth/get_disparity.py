import numpy as np
import pyzed.sl as sl
import cv2
from camera_utils import get_camera_intrinsics
import compute_utils
from pcd_utils import update_point_cloud
import open3d as o3d

number_of_image_channels = 3

min_disp = 0  # 最小视差值
num_disp = 96  # 视差范围
blockSize = 6  # 每个像素周围的块大小
P1 = 8 * number_of_image_channels * blockSize  # 控制低纹理区域的平滑程度
P2 = 32 * number_of_image_channels * blockSize  # 控制高纹理区域的平滑程度
disp12MaxDiff = 1  # 最大视差差
preFilterCap = 63  # 映射滤波器大小
uniquenessRatio = 10  # 唯一性比率，判断是否存在唯一的匹配，值越小，唯一性匹配的要求越高
speckleWindowSize = 100  # 斑点窗口大小
speckleRange = 10  # 斑点范围
mode = cv2.StereoSGBM_MODE_SGBM

# 初始化相机
zed = sl.Camera()
init = sl.InitParameters()
init.depth_mode = sl.DEPTH_MODE.ULTRA
init.coordinate_units = sl.UNIT.MILLIMETER
init.camera_resolution = sl.RESOLUTION.VGA

# 启动相机
status = zed.open(init)
runtime_params = sl.RuntimeParameters()

# 获取相机内参，焦距和基线距离
camera_intrinsics, focal_left_x, baseline_mm = get_camera_intrinsics(zed)

left = sl.Mat()
right = sl.Mat()

if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
    # 获取左,右图像
    zed.retrieve_image(left, sl.VIEW.LEFT)
    zed.retrieve_image(right, sl.VIEW.RIGHT)

    left_data = left.get_data()
    right_data = right.get_data()

    # 执行视差计算
    disp = compute_utils.compute_disparity(left=left_data, right=right_data,
                                           min_disp=min_disp, num_disp=num_disp,
                                           block_size=blockSize,
                                           P1=P1, P2=P2,
                                           disp12MaxDiff=disp12MaxDiff,
                                           preFilterCap=preFilterCap,
                                           uniquenessRatio=uniquenessRatio,
                                           speckleWindowSize=speckleWindowSize,
                                           speckleRange=speckleRange,
                                           mode=mode)

    # 定义中值滤波的核大小和双边滤波的参数
    median_kernel_size = 5
    # bilateral_diameter = 9
    # bilateral_sigma_color = 75
    # bilateral_sigma_space = 75

    # 对视差图进行中值滤波处理
    disparity_map_median = cv2.medianBlur(disp, median_kernel_size)

    # 对中值滤波后的视差图进行双边滤波处理
    # disparity_map_filtered = cv2.bilateralFilter(disparity_map_median, bilateral_diameter, bilateral_sigma_color,
    #                                              bilateral_sigma_space)

    # 可视化视差图
    # disp_8U = cv2.normalize(disp, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    disp_0_1 = np.clip((disp - min_disp) / num_disp, 0, 1)

    depth = compute_utils.compute_depth(disp, baseline_mm, focal_left_x)
    depth_map_0_1 = cv2.normalize(depth, None, 0, 1, cv2.NORM_MINMAX)

    print("Minimum and maximum depth is: ", np.min(depth), np.max(depth))

    # cv2.imshow('left', left_data)
    # cv2.imshow('right', right_data)
    cv2.imshow('disparity_0_1', disp_0_1)
    cv2.imshow('disparity', disp)
    cv2.imshow('disparity_map_median', disparity_map_median)
    cv2.imshow('disparity_map_filtered', disparity_map_filtered)
    # cv2.imshow('depth', depth)
    # cv2.imshow('depth_map_0_1', depth_map_0_1)

    cv2.imwrite('image/kochi/left.png', left_data)
    cv2.imwrite('image/kochi/right.png', right_data)
    cv2.imwrite('image/kochi/disparity.png', disp)
    cv2.imwrite('image/kochi/kochi_depth.png', depth)

    cv2.waitKey()
    cv2.destroyAllWindows()
# # 点云计算
# new_pcd = compute_utils.depth2pcd_with_o3d(depth, camera_intrinsics)
# new_points = np.asarray(new_pcd.points)
#
# # 点云对象和可视化窗口
# pcd = o3d.geometry.PointCloud()
# vis = o3d.visualization.Visualizer()
# vis.create_window()
# vis.add_geometry(new_pcd)
#
# update_point_cloud(pcd, vis, new_points)
#
# print("Point Cloud contains: ", len(pcd.points))
#
# cv2.waitKey()
# cv2.destroyAllWindows()
#
# o3d.io.write_point_cloud("../result/pcd/teddy.pcd", pcd)
