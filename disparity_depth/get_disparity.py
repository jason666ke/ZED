import numpy as np
import pyzed.sl as sl
import cv2
from camera_utils import get_camera_intrinsics
import compute_utils
from pcd_utils import update_point_cloud
import open3d as o3d

left = cv2.imread('image/teddy-png-2/teddy/im2.png')
right = cv2.imread('image/teddy-png-2/teddy/im6.png')

if left is None or right is None:
    print("Could not read the image files")
    exit()

min_disp = 0  # 最小视差值
num_disp = 96  # 视差范围
blockSize = 6  # 每个像素周围的块大小
uniquenessRatio = 10  # 唯一性比率，判断是否存在唯一的匹配，值越小，唯一性匹配的要求越高
speckleRange = 0  # 斑点范围
speckleWindowSize = 0  # 斑点窗口大小
disp12MaxDiff = -1  # 最大视差差
P1 = 8 * 3 * blockSize  # 控制低纹理区域的平滑程度
P2 = 32 * 3 * blockSize  # 控制高纹理区域的平滑程度

# 创建一个StereoSGBM实例
stereo = cv2.StereoSGBM.create(
    minDisparity=min_disp,
    numDisparities=num_disp,
    blockSize=blockSize,
    uniquenessRatio=uniquenessRatio,
    speckleRange=speckleRange,
    speckleWindowSize=speckleWindowSize,
    disp12MaxDiff=disp12MaxDiff,
    P1=P1,
    P2=P2
)

# SGBM算法计算出的视差值为整数形式，需要进行转化得到真实视差值
disp = stereo.compute(left, right).astype(np.float32) / 16.0
# disp = stereo.compute(left, right)
# 视差图归一化到 [0, 1]的范围内
disp_0_1 = cv2.normalize(disp, None, 0, 1, cv2.NORM_MINMAX)

disp_8U = cv2.normalize(disp, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

print("Minimum and maximum disparity is: ", np.min(disp), np.max(disp))
print("Minimum and maximum disparity_8U is: ", np.min(disp_8U), np.max(disp_8U))

# 初始化相机
zed = sl.Camera()
init = sl.InitParameters()
init.depth_mode = sl.DEPTH_MODE.ULTRA
init.coordinate_units = sl.UNIT.MILLIMETER
init.camera_resolution = sl.RESOLUTION.VGA

# 启动相机
status = zed.open(init)

# 获取相机内参，焦距和基线距离
camera_intrinsics, focal_left_x, baseline_mm = get_camera_intrinsics(zed)

# 计算深度
# depth = np.zeros_like(disp)
# for row in range(disp.shape[0]):
#     for col in range(disp.shape[1]):
#         if disp[row, col] != 0:
#             depth[row, col] = (focal_left_x * baseline_mm) / disp[row, col]
depth = compute_utils.compute_depth(disp, baseline_mm, focal_left_x)
cv2.imwrite('image/teddy-png-2/teddy/teddy_depth.png', depth)

print("Minimum and maximum depth is: ", np.min(depth), np.max(depth))

cv2.imshow('disparity_0_1', disp_0_1)
cv2.imshow('disparity_8U', disp_8U)
cv2.imshow('depth', depth)

# 点云计算
new_pcd = compute_utils.depth2pcd_with_o3d(depth, camera_intrinsics)
new_points = np.asarray(new_pcd.points)

# 点云对象和可视化窗口
pcd = o3d.geometry.PointCloud()
vis = o3d.visualization.Visualizer()
vis.create_window()
vis.add_geometry(new_pcd)

update_point_cloud(pcd, vis, new_points)

print("Point Cloud contains: ", len(pcd.points))

cv2.waitKey()
cv2.destroyAllWindows()

o3d.io.write_point_cloud("../result/pcd/teddy.pcd", pcd)