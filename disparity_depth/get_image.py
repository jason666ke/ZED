import os
import threading
import time
from time import sleep

import numpy as np
import pyzed.sl as sl
import cv2
import open3d as o3d

# 创建保存图像的文件夹
save_folder = "image"
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# 定义全局变量用以判断相机状态
camera_status = False
exit_program = False

"""
    SGBM算法参数设置
"""
window_size = 5
min_disp = 16  # 最小视差值
num_disp = 192 - min_disp  # 视差范围
blockSize = window_size  # 每个像素周围的块大小
uniquenessRatio = 1  # 唯一性比率，判断是否存在唯一的匹配，值越小，唯一性匹配的要求越高
speckleRange = 3  # 斑点范围
speckleWindowSize = 3  # 斑点窗口大小
disp12MaxDiff = 200  # 最大视差差
P1 = 600  # 控制低纹理区域的平滑程度
P2 = 2400  # 控制高纹理区域的平滑程度


# 使用双目摄像头进行视差计算
def compute_disparity(left, right):
    global camera_status
    if not camera_status:
        print("Camera status wrong!")
        return None

    # 创建一个StereoSGBM实例
    stereo = cv2.StereoSGBM.create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=window_size,
        uniquenessRatio=uniquenessRatio,
        speckleRange=speckleRange,
        speckleWindowSize=speckleWindowSize,
        disp12MaxDiff=disp12MaxDiff,
        P1=P1,
        P2=P2
    )

    # 将整数的视差值转化为浮点数，便于更好的表示
    disp = stereo.compute(left, right).astype(np.float32) / 16.0
    disp_normalize = (disp - min_disp) / num_disp
    return disp_normalize


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

    return depth_map


def depth2pcd(depth_map, camera_intrinsics, flatten=False):
    fx, fy, cx, cy = camera_intrinsics
    height, width = np.mgrid[0:depth_map.shape[0], 0:depth_map.shape[1]]

    z = depth_map
    x = (width - cx) * z / fx
    y = (width - cy) * z / fy

    xyz = np.dstack((x, y, z)) if flatten is False else np.dstack((x, y, z)).reshape(-1, 3)
    return xyz


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


def show_point_cloud(point_cloud):
    pcd = o3d.io.read_point_cloud(point_cloud)
    o3d.visualization.draw([point_cloud])


# 图像处理线程
def image_processing_thread():
    global camera_status, exit_program

    zed = sl.Camera()
    init = sl.InitParameters()
    init.depth_mode = sl.DEPTH_MODE.ULTRA
    init.camera_resolution = sl.RESOLUTION.HD720

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

    # 获取摄像头的标定参数
    camera_info = zed.get_camera_information()
    calibration_params = camera_info.camera_configuration.calibration_parameters
    # sensor_width_pixel = calibration_params.left_cam.image_size.width  # 左摄像头获取图像的像素宽度
    # sensor_height_pixel = calibration_params.left_cam.image_size.height  # 左摄像头获取图像的像素高度
    # focal_left_x = calibration_params.left_cam.fx  # 焦距（像素单位）
    focal_left_metric = calibration_params.left_cam.focal_length_metric  # real focal length in millimeters
    print("Left cam fx: {0} millimeters".format(focal_left_metric))
    baseline_mm = calibration_params.get_camera_baseline()  # 基线距离(毫米为单位)
    # baseline_pixel = (baseline_mm * focal_left_x) / focal_left_metric  # 基线距离（像素为单位）
    print("Baseline: {0} millimeters".format(baseline_mm))
    fx = calibration_params.left_cam.fx
    fy = calibration_params.left_cam.fy
    cx = calibration_params.left_cam.cx
    cy = calibration_params.left_cam.cy
    camera_intrinsics = [fx, fy, cx, cy]

    while camera_status:
        if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
            # 获取左,右图像
            zed.retrieve_image(left, sl.VIEW.LEFT)
            zed.retrieve_image(right, sl.VIEW.RIGHT)

            left_data = left.get_data()
            right_data = right.get_data()

            # 执行视差计算
            disp = compute_disparity(left_data, right_data)

            # 执行深度计算
            depth = compute_depth(disp, baseline_mm, focal_left_metric)

            # 点云计算
            pcd = depth2pcd(depth, camera_intrinsics)

            # 可视化
            if disp is not None:
                # cv2.imshow("left", left_data)
                # cv2.imshow("right", right_data)
                cv2.imshow("disparity", disp)
                cv2.imshow("depth", depth)
                # disp_and_depth = np.hstack([disp, depth])
                # cv2.imshow("disparity and depth", disp_and_depth)
                # show_point_cloud(pcd)

            key = cv2.waitKey(10)
            if key == ord('q') or key == ord('Q'):
                exit_program = True
                print("Exit camera thread: ", exit_program)
                break

    zed.close()


# 主程序
if __name__ == "__main__":
    camera_thread = threading.Thread(target=image_processing_thread)
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
