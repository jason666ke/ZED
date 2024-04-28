import numpy as np
import open3d as o3d
import cv2
from disparity_depth import compute_utils
from disparity_depth.pcd_utils import update_point_cloud
import pyzed.sl as sl
from disparity_depth import camera_utils
zed = sl.Camera()
init = sl.InitParameters()
init.depth_mode = sl.DEPTH_MODE.ULTRA
# init.camera_resolution = sl.RESOLUTION.HD720
init.camera_resolution = sl.RESOLUTION.VGA

status = zed.open(init)
if status != sl.ERROR_CODE.SUCCESS:
    print(repr(status))
    camera_status = False
else:
    print("Camera open successfully!")
    camera_status = True

runtime_params = sl.RuntimeParameters()

left = sl.Mat()
right = sl.Mat()

# 获取摄像头内参矩阵，焦距和基线距离
camera_intrinsics, focal_left_x, baseline_mm = camera_utils.get_camera_intrinsics(zed)
image_size = camera_utils.get_image_size(zed)

# 点云对象和可视化窗口
pcd = o3d.geometry.PointCloud()
points = np.random.rand(10000, 3)
pcd.points = o3d.utility.Vector3dVector(points)

vis = o3d.visualization.Visualizer()
vis.create_window()
vis.add_geometry(pcd)

model = compute_utils.load_model()

if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
    # 获取左,右图像
    zed.retrieve_image(left, sl.VIEW.LEFT)
    zed.retrieve_image(right, sl.VIEW.RIGHT)

    left_data = left.get_data()
    right_data = right.get_data()

    # compute disparity
    disp = compute_utils.compute_disparity_CRE(left_data, right_data, model)

    # 计算深度、点云并可视化
    if disp is not None:
        # 执行深度计算
        depth = compute_utils.compute_depth(disp, baseline_mm, focal_left_x)
        # Compute point cloud
        new_point_cloud = compute_utils.depth2pcd_with_o3d(left_data, depth, camera_intrinsics)
        pcd.points = new_point_cloud.points
        pcd.colors = new_point_cloud.colors

        # visualization
        # disp_vis = compute_utils.img_visualize(disp)
        # depth_vis = compute_utils.img_visualize(depth)
        # both_view = np.hstack([left_data, right_data])
        # cv2.imshow("Left and Right", both_view)
        # cv2.imshow("Disparity and depth", np.hstack([disp_vis, depth_vis]))
        vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()
        # vis.run()
        # vis.update_geometry(pcd)

        # o3d.visualization.draw_geometries([new_point_cloud])

