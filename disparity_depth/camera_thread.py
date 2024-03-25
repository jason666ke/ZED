import os
import threading
from time import sleep

import cv2
import numpy as np
import open3d as o3d
import pyzed.sl as sl

import compute_utils
from camera_utils import get_camera_intrinsics
from real_time_pcd import update_point_cloud

# 创建保存图像的文件夹
save_folder = "image"
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# 定义全局变量用以判断相机状态
camera_status = False
exit_program = False


# 图像处理线程
def image_processing_thread(min_disp, num_disp, block_size, uniquenessRatio, speckleRange, speckleWindowSize,
                            disp12MaxDiff, P1, P2, mode):
    global camera_status, exit_program

    zed = sl.Camera()
    init = sl.InitParameters()
    init.depth_mode = sl.DEPTH_MODE.ULTRA
    # init.camera_resolution = sl.RESOLUTION.HD720
    init.camera_resolution = sl.RESOLUTION.VGA

    status = zed.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        print(repr(status))
        camera_status = False
        return
    else:
        print("Camera open successfully!")
        camera_status = True

    runtime_params = sl.RuntimeParameters()

    left = sl.Mat()
    right = sl.Mat()

    # 获取摄像头内参矩阵，焦距和基线距离
    camera_intrinsics, focal_left_x, baseline_mm = get_camera_intrinsics(zed)

    # 点云对象和可视化窗口
    pcd = o3d.geometry.PointCloud()
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    while camera_status:
        if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
            # 获取左,右图像
            zed.retrieve_image(left, sl.VIEW.LEFT)
            zed.retrieve_image(right, sl.VIEW.RIGHT)

            left_data = left.get_data()
            right_data = right.get_data()

            # 执行视差计算
            # disp = compute_disparity(left_data, right_data, min_disp, num_disp, block_size, uniquenessRatio,
            #                          speckleRange, speckleWindowSize, disp12MaxDiff, P1, P2, mode)
            disp = compute_utils.compute_disparity(left_data, right_data, min_disp, num_disp, block_size,
                                                   uniquenessRatio,
                                                   speckleRange, speckleWindowSize, disp12MaxDiff, P1, P2, mode)

            # 可视化视差图
            # disp_8U = cv2.normalize(disp, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            disp_0_1 = np.clip((disp - min_disp) / num_disp, 0, 1)

            # 计算深度、点云并可视化
            if disp is not None:
                print("Min and max disparity: ", np.min(disp), np.max(disp))

                # 执行深度计算
                # depth = compute_depth(disp, baseline_mm, focal_left_x)
                depth = compute_utils.compute_depth(disp, baseline_mm, focal_left_x)
                print("Minimum and maximum depth is: ", np.min(depth), np.max(depth))

                # 点云计算
                # new_pcd = depth2pcd_with_o3d(depth, camera_intrinsics)
                new_pcd = compute_utils.depth2pcd_with_o3d(depth, camera_intrinsics)
                new_points = np.asarray(new_pcd.points)

                update_point_cloud(pcd, vis, new_points)

                # 可视化
                both_view = np.hstack([left_data, right_data])
                cv2.imshow("Left and Right", both_view)
                disp_and_depth = np.hstack([disp_0_1, depth])
                cv2.imshow("Disp and Depth", disp_and_depth)

            key = cv2.waitKey(10)
            if key == ord('q') or key == ord('Q'):
                exit_program = True
                print("Exit camera thread: ", exit_program)
                break

    zed.close()


# 主程序
if __name__ == "__main__":
    # parameters for StereoSGBM

    # min_disp = 16  # 最小视差值
    # num_disp = 192 - min_disp  # 视差范围
    # window_size = 5
    # blockSize = window_size  # 每个像素周围的块大小
    # uniquenessRatio = 1  # 唯一性比率，判断是否存在唯一的匹配，值越小，唯一性匹配的要求越高
    # speckleRange = 3  # 斑点范围
    # speckleWindowSize = 3  # 斑点窗口大小
    # disp12MaxDiff = 200  # 最大视差差
    # P1 = 600  # 控制低纹理区域的平滑程度
    # P2 = 2400  # 控制高纹理区域的平滑程度
    number_of_image_channels = 3

    min_disp = 1  # 最小视差值
    num_disp = 96  # 视差范围
    blockSize = 6  # 每个像素周围的块大小
    uniquenessRatio = 10  # 唯一性比率，判断是否存在唯一的匹配，值越小，唯一性匹配的要求越高
    speckleRange = 0  # 斑点范围
    speckleWindowSize = 0  # 斑点窗口大小
    disp12MaxDiff = -1  # 最大视差差
    P1 = 8 * number_of_image_channels * blockSize  # 控制低纹理区域的平滑程度
    P2 = 32 * number_of_image_channels * blockSize  # 控制高纹理区域的平滑程度
    mode = cv2.StereoSGBM_MODE_SGBM_3WAY

    # camera start
    camera_thread = threading.Thread(target=image_processing_thread, args=(
        min_disp, num_disp, blockSize, uniquenessRatio, speckleRange, speckleWindowSize, disp12MaxDiff, P1, P2, mode
    ))
    print("Camera thread start...")
    camera_thread.start()

    # while True:
    #     # 检测键盘输入
    #     key = cv2.waitKey(10)
    #     if key == ord('q') or key == ord('Q'):
    #         exit_program = True
    #     if exit_program:
    #         break
    while True:
        if exit_program:
            break
        # print("Running main thread for 5s...")
        sleep(5)
        pass

    print("Exit main thread: ", exit_program)
    # 关闭相机和窗口
    camera_status = False
    camera_thread.join()
    print("Camera thread end.")

    exit(0)
