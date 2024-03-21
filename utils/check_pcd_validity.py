import numpy as np


def check_point_cloud_validity(point_cloud_data):
    xyz = point_cloud_data[:, :, :3]

    valid_xyz = ~np.isnan(xyz).any(axis=2) & ~np.isinf(xyz).any(axis=2)

    colors = point_cloud_data[:, :, 3]  # 多维数组中通过索引提取某一个特定维度时会失去那个维度
    valid_colors = ~np.isnan(colors) & ~np.isinf(colors)

    valid_points = valid_xyz & valid_colors
    num_valid_points = np.sum(valid_points)

    is_empty = (num_valid_points == 0)

    return {
        "is empty": is_empty,
        "num_valid_points": num_valid_points,
        "total points": point_cloud_data.shape[0] * point_cloud_data.shape[1],
        "validity_percentage": (num_valid_points / (point_cloud_data.shape[0] * point_cloud_data.shape[1])) * 100
    }
