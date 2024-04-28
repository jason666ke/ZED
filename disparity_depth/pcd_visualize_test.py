import numpy as np
import open3d as o3d
from disparity_depth import pcd_utils

height = 376
width = 624
pcd = o3d.geometry.PointCloud()
points = np.random.rand(10000, 3)
pcd.points = o3d.utility.Vector3dVector(points)

vis = o3d.visualization.Visualizer()
vis.create_window('PCD Visualization', height, width)
vis.add_geometry(pcd)

while True:
    new_points = np.random.rand(10000, 3)
    pcd_utils.update_point_cloud(pcd, vis, new_points)
