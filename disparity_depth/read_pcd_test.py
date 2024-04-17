import open3d as o3d

pcd_path = "../result/teddy.pcd"
pcd = o3d.io.read_point_cloud(pcd_path)
o3d.visualization.draw_geometries([pcd])