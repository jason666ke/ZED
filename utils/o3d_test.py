import open3d as o3d
import numpy as np

print("读取点云并可视化")
pcd = o3d.io.read_point_cloud("../pcd_package/灯.pcd")
print(pcd)
print(np.asarray(pcd.points))
o3d.visualization.draw_geometries([pcd])

