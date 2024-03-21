import open3d as o3d
import numpy as np
import time

def update_point_cloud(pcd, vis, new_points):
    """
    Update the point cloud
    :param pcd: point cloud data
    :param vis: pcd visualizer
    :return:
    """
    # test
    # new_points = np.random.rand(10000, 3)

    pcd.points = o3d.utility.Vector3dVector(new_points)

    vis.update_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()


if __name__ == '__main__':
    pcd = o3d.geometry.PointCloud()
    points = np.random.rand(10000, 3)
    pcd.points = o3d.utility.Vector3dVector(points)

    vis = o3d.visualization.Visualizer()
    vis.create_window()

    vis.add_geometry(pcd)

    while True:
        new_points = np.random.rand(10000, 3)
        update_point_cloud(pcd, vis, new_points)
        time.sleep(0.2)




