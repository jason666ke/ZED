import open3d as o3d
import numpy as np
import time


def update_point_cloud(pcd, vis, new_points):
    """
    Update the point cloud
    :param pcd: point cloud object
    :param vis: pcd visualizer
    :param new_points: new point cloud data
    :return:
    """
    # test
    # new_points = np.random.rand(10000, 3)

    pcd.points = o3d.utility.Vector3dVector(new_points)

    vis.update_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()


def get_pcd_info(point_cloud):
    # 打印点云的数量
    num_points = len(point_cloud.points)
    print("Point cloud contains", num_points, "points.")

    # 检查是否包含颜色信息
    if point_cloud.has_colors():
        # 获取颜色数组
        colors = np.asarray(point_cloud.colors)
        # 打印颜色数组的形状和前几个颜色值
        print("Point cloud has color information.")
        print("Color array shape:", colors.shape)
        print("First few color values:")
        print(colors[:5])  # 打印前五个颜色值
    else:
        print("Point cloud does not have color information.")


if __name__ == '__main__':
    pcd = o3d.geometry.PointCloud()
    # 需要一开始初始化一次points属性，不然无法可视化
    points = np.random.rand(10000, 3)
    pcd.points = o3d.utility.Vector3dVector(points)

    vis = o3d.visualization.Visualizer()
    vis.create_window()

    vis.add_geometry(pcd)

    while True:
        get_pcd_info(pcd)
        new_points = np.random.rand(10000, 3)
        update_point_cloud(pcd, vis, new_points)
        time.sleep(0.2)
