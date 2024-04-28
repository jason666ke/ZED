import numpy as np
import open3d as o3d
import pandas as pd

pcd = o3d.io.read_point_cloud("./test_data/test_point_cloud.pcd")
# pcd = o3d.geometry.PointCloud()
# data = pd.read_csv("kochi_pcd.csv", index_col=0)
#
# xyz = data.iloc[:, :3].values.astype(np.float32)
# normals = data.iloc[:, 3:6].values.astype(np.float32)
# colors = (data.iloc[:, 6:].values / 255.0).astype(np.float32)
#
# pcd.points = o3d.utility.Vector3dVector(xyz)
# pcd.normals = o3d.utility.Vector3dVector(normals)
# pcd.colors = o3d.utility.Vector3dVector(colors)
print(pcd)
o3d.visualization.draw_geometries([pcd])