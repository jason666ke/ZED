import cv2
import numpy as np

# 读取左右相机的图像
left_image = cv2.imread('image/kochi/left.png', cv2.IMREAD_GRAYSCALE)
right_image = cv2.imread('image/kochi/right.png', cv2.IMREAD_GRAYSCALE)

# 定义SGBM算法的参数
min_disp = 0
max_disp = 64  # 视差范围
block_size = 5  # 匹配块的大小
P1 = 8 * 3 * block_size**2  # 控制视差图平滑度的参数
P2 = 32 * 3 * block_size**2  # 控制视差图平滑度的参数
disp12_max_diff = 1  # 左右视差图之间的最大差异
pre_filter_cap = 63  # 预处理滤波器的截断值
uniqueness_ratio = 10  # 唯一性约束
speckle_window_size = 100  # 滤除小连通区域的窗口大小
speckle_range = 32  # 斑点滤波器的范围

# 创建SGBM对象并计算左视图的视差图
sgbm = cv2.StereoSGBM_create(
    minDisparity=min_disp,
    numDisparities=max_disp,
    blockSize=block_size,
    P1=P1,
    P2=P2,
    disp12MaxDiff=disp12_max_diff,
    preFilterCap=pre_filter_cap,
    uniquenessRatio=uniqueness_ratio,
    speckleWindowSize=speckle_window_size,
    speckleRange=speckle_range
)
left_disp = sgbm.compute(left_image, right_image)
left_disp = cv2.imread("image/kochi/disparity.png")
# 创建右视图匹配器对象
# right_matcher = cv2.ximgproc.createRightMatcher(sgbm)
#
# 计算右视图的视差图
# right_disp = right_matcher.compute(right_image, left_image)

# 使用视差图的有效范围进行校正
# 将视差值归一化到 [0, 255]
normalized_left_disp = cv2.normalize(src=left_disp, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
# normalized_right_disp = cv2.normalize(src=right_disp, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

# 可选：将左右视差图合并为最终视差图
# disparity_map = cv2.addWeighted(src1=normalized_left_disp, alpha=0.5, src2=normalized_right_disp, beta=0.5, gamma=0)

# 定义相机内参和基线长度
fx = 260.2951  # 焦距
baseline = 119.84703  # 基线长度

# 计算深度图
depth_map = (fx * baseline) / left_disp

# 可选：对深度图进行可视化或保存
cv2.imshow('Depth Map', depth_map)

# 可选：对最终视差图进行可视化或保存
cv2.imshow('Disparity Map', normalized_left_disp)
cv2.waitKey(0)
cv2.destroyAllWindows()
