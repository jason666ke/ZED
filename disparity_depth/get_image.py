import os
import threading
import time
from time import sleep

import numpy as np
import pyzed.sl as sl
import cv2
import open3d as o3d
from real_time_pcd import update_point_cloud

# 创建保存图像的文件夹
save_folder = "image"
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# 定义全局变量用以判断相机状态
camera_status = False
exit_program = False


# 使用双目摄像头进行视差计算
def compute_disparity(left, right, min_disp, num_disp, block_size,
                      uniquenessRatio, speckleRange, speckleWindowSize,
                      disp12MaxDiff, P1, P2):
    # 创建一个StereoSGBM实例
    stereo = cv2.StereoSGBM.create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=block_size,
        uniquenessRatio=uniquenessRatio,
        speckleRange=speckleRange,
        speckleWindowSize=speckleWindowSize,
        disp12MaxDiff=disp12MaxDiff,
        P1=P1,
        P2=P2
    )

    # 将整数的视差值转化为浮点数，便于更好的表示
    disp = stereo.compute(left, right).astype(np.float32) / 16.0
    # 视差图归一化到 [0, 1]的范围内
    disp = np.clip((disp - min_disp) / num_disp, 0, 1)

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

    return depth_map


def depth2pcd(depth_map, camera_intrinsics, flatten=False):
    # todo: 加入上色代码
    fx, fy, cx, cy = camera_intrinsics
    height, width = np.mgrid[0:depth_map.shape[0], 0:depth_map.shape[1]]

    z = depth_map
    x = (width - cx) * z / fx
    y = (height - cy) * z / fy

    xyz = np.dstack((x, y, z)) if flatten is False else np.dstack((x, y, z)).reshape(-1, 3)

    return xyz


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


# 图像处理线程
def image_processing_thread(min_disp, num_disp, block_size, uniquenessRatio, speckleRange, speckleWindowSize,
                            disp12MaxDiff, P1, P2):
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

    # 获取摄像头的标定参数
    camera_info = zed.get_camera_information()
    calibration_params = camera_info.camera_configuration.calibration_parameters

    # 焦距和基线距离
    focal_left_x = calibration_params.left_cam.fx  # 焦距（像素单位）
    baseline_mm = calibration_params.get_camera_baseline()  # 基线距离(毫米为单位)
    print("Left cam fx: {0} pixel".format(focal_left_x))
    print("Baseline: {0} millimeters".format(baseline_mm))

    # 摄像头内参
    width = calibration_params.left_cam.image_size.width
    height = calibration_params.left_cam.image_size.height
    fx = calibration_params.left_cam.fx
    fy = calibration_params.left_cam.fy
    cx = calibration_params.left_cam.cx
    cy = calibration_params.left_cam.cy
    camera_intrinsics = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

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
            disp = compute_disparity(left_data, right_data, min_disp, num_disp, block_size, uniquenessRatio,
                                     speckleRange, speckleWindowSize, disp12MaxDiff, P1, P2)

            # 计算深度、点云并可视化
            if disp is not None:
                # 执行深度计算
                depth = compute_depth(disp, baseline_mm, focal_left_x)

                # 点云计算
                new_pcd = depth2pcd_with_o3d(depth, camera_intrinsics)
                new_points = np.asarray(new_pcd.points)

                update_point_cloud(pcd, vis, new_points)

                # 可视化
                both_view = np.hstack([left_data, right_data])
                cv2.imshow("Left and Right", both_view)
                disp_and_depth = np.hstack([disp, depth])
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

    min_disp = 16  # 最小视差值
    num_disp = 192 - min_disp  # 视差范围
    window_size = 5
    blockSize = window_size  # 每个像素周围的块大小
    uniquenessRatio = 1  # 唯一性比率，判断是否存在唯一的匹配，值越小，唯一性匹配的要求越高
    speckleRange = 3  # 斑点范围
    speckleWindowSize = 3  # 斑点窗口大小
    disp12MaxDiff = 200  # 最大视差差
    P1 = 600  # 控制低纹理区域的平滑程度
    P2 = 2400  # 控制高纹理区域的平滑程度

    # camera start
    camera_thread = threading.Thread(target=image_processing_thread, args=(
        min_disp, num_disp, blockSize, uniquenessRatio, speckleRange, speckleWindowSize, disp12MaxDiff, P1, P2
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
