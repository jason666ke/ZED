import numpy as np
import open3d as o3d
import cv2
from disparity_depth import compute_utils
from disparity_depth.pcd_utils import update_point_cloud

depth = cv2.imread("image/kochi/kochi_depth.png", cv2.IMREAD_UNCHANGED)

fx = 100
fy = 100
cx = depth.shape[1] // 2
cy = depth.shape[0] // 2

width = depth.shape[1]
height = depth.shape[0]

camera_intrinsics = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

# Compute point cloud
point_cloud = compute_utils.depth2pcd(depth, camera_intrinsics)

# 点云对象和可视化窗口
pcd = o3d.geometry.PointCloud()
# points = np.random.rand(10000, 3)
# pcd.points = o3d.utility.Vector3dVector(points)

vis = o3d.visualization.Visualizer()
vis.create_window('PCD Visualization', height, width)
vis.add_geometry(pcd)

update_point_cloud(pcd, vis, point_cloud)

