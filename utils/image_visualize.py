import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
# 实时更新函数
def update_point_cloud(fig, data):
    ax = fig.add_subplot(111, projection='3d')

    # 清除之前的数据点
    ax.clear()

    # 更新点云数据
    ax.scatter(data[:, :, 0], data[:, :, 1], data[:, :, 2])

    # 设置坐标轴标签
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    # 设置图形标题
    ax.set_title('Real-time Point Cloud Visualization')

    # 重新绘制
    plt.draw()
    plt.pause(0.01)  # 添加短暂延迟以便图形更新

