import os
import threading

import numpy as np
import pyzed.sl as sl
import cv2

# 创建保存图像的文件夹
save_folder = "image"
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# 定义全局变量用以判断相机状态
camera_status = False

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


# # 深度图
# def compute_depth(disparity_map):
#     # 摄像机参数
#     baseline = 0.5  # 基线距离
#     # return baseline in unit defined in sl.InitParameters.coordinate_units
#     focal_length = sl.CalibrationParameters.get_camera_baseline()
#
#     # 根据视差图计算深度图
#     depth_map = (focal_length * baseline) / disparity_map
#
#     return depth_map


# 图像处理线程
def image_processing_thread():
    global camera_status

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

    while camera_status:
        if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
            # 获取左,右图像
            zed.retrieve_image(left, sl.VIEW.LEFT)
            zed.retrieve_image(right, sl.VIEW.RIGHT)

            left_data = left.get_data()
            right_data = right.get_data()

            # 执行视差计算
            disp = compute_disparity(left_data, right_data)

            # 可视化
            if disp is not None:
                cv2.imshow("left", left_data)
                cv2.imshow("right", right_data)
                cv2.imshow("disparity", disp)

            key = cv2.waitKey(10)
            if key == ord('q') or key == ord('Q'):
                break

    zed.close()


# 主程序
if __name__ == "__main__":
    camera_thread = threading.Thread(target=image_processing_thread)
    camera_thread.start()

    while True:
        # 检测键盘输入
        key = cv2.waitKey(10)
        if key == ord('q') or key == ord('Q'):
            break

    # 关闭相机和窗口
    camera_status = False
    camera_thread.join()
    cv2.destroyAllWindows()

    exit(0)
