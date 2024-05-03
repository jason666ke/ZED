import numpy as np
import cv2
import compute_utils
from skimage.metrics import structural_similarity


def get_disp(disp_path):
    disp = cv2.imread(disp_path, cv2.IMREAD_UNCHANGED)
    return disp.astype(np.float32) / 32


def compute_mse(gt_disparity, predicted_disparity):
    """
    计算均方误差（MSE）
    """
    mse = np.mean((gt_disparity - predicted_disparity) ** 2)
    return mse


def compute_mae(gt_disparity, predicted_disparity):
    """
    计算平均绝对误差（MAE）
    """
    mae = np.mean(np.abs(gt_disparity - predicted_disparity))
    return mae


def compute_ssim(gt_disparity, predicted_disparity):
    """
    计算结构相似性指数（SSIM）
    """
    score = structural_similarity(gt_disparity, predicted_disparity,
                                  data_range=predicted_disparity.max() - predicted_disparity.min())
    return score


def compute_epe(gt_disparity, predicted_disparity):
    """
    计算端点误差（End-point Error）
    """
    epe = np.sqrt((gt_disparity - predicted_disparity) ** 2)
    return epe


def evaluate_disparity(gt_disparity, predicted_disparity):
    """
    评估视差图质量
    """
    # 读取视差图
    # gt_disparity = cv2.imread(gt_disparity_path, cv2.IMREAD_GRAYSCALE)
    # predicted_disparity = cv2.imread(predicted_disparity_path, cv2.IMREAD_GRAYSCALE)

    # 计算评估指标
    mse = compute_mse(gt_disparity, predicted_disparity)
    # mae = compute_mae(gt_disparity, predicted_disparity)
    epe = compute_epe(gt_disparity, predicted_disparity)
    ssim = compute_ssim(gt_disparity, predicted_disparity)

    # 打印结果
    print("Mean Squared Error (MSE):", mse)
    # print("Mean Absolute Error (MAE):", mae)
    print("End-point Error (EPE):", np.mean(epe))
    print("Structural Similarity Index (SSIM):", ssim)


if __name__ == "__main__":
    left = cv2.imread("../data/cones-png-2/cones/im2.png")
    right = cv2.imread("../data/cones-png-2/cones/im6.png")
    print("left, right shape: {0}, {1}".format(left.shape, right.shape))
    # load CREStereo model
    model = compute_utils.load_model()

    # 输入真实视差图和预测视差图的路径
    gt_disparity_path = "../data/cones-png-2/cones/disp2.png"
    gt_disparity = get_disp(gt_disparity_path)
    gt_vis = compute_utils.img_visualize(gt_disparity)

    # default SGBM
    # stereo_SGBM = cv2.StereoSGBM.create()
    # pred_disp_SGBM = stereo_SGBM.compute(left, right)
    # pred_vis_SGBM = compute_utils.img_visualize(pred_disp_SGBM)

    # good params SGBM
    number_of_image_channels = 3
    min_disp = 0  # 最小视差值
    num_disp = 96  # 视差范围
    blockSize = 6  # 每个像素周围的块大小
    P1 = 8 * number_of_image_channels * blockSize  # 控制低纹理区域的平滑程度
    P2 = 32 * number_of_image_channels * blockSize  # 控制高纹理区域的平滑程度
    disp12MaxDiff = 1  # 最大视差差
    preFilterCap = 63  # 映射滤波器大小
    uniquenessRatio = 10  # 唯一性比率，判断是否存在唯一的匹配，值越小，唯一性匹配的要求越高
    speckleWindowSize = 100  # 斑点窗口大小
    speckleRange = 10  # 斑点范围
    mode = cv2.StereoSGBM_MODE_SGBM
    stereo_SGBM = cv2.StereoSGBM.create(
        minDisparity=min_disp, numDisparities=num_disp, blockSize=blockSize,
        P1=P1, P2=P2, disp12MaxDiff=disp12MaxDiff, preFilterCap=preFilterCap,
        uniquenessRatio=uniquenessRatio, speckleRange=speckleRange, mode=mode
    )
    pred_disp_SGBM = stereo_SGBM.compute(left, right)
    pred_vis_SGBM = compute_utils.img_visualize(pred_disp_SGBM)

    predicted_disparity_CRE = compute_utils.compute_disparity_CRE(left, right, model) / 8
    pred_vis_CRE = compute_utils.img_visualize(predicted_disparity_CRE)

    # 评估视差图
    print("SGBM: gt and pred shape: {0}, {1}".format(gt_disparity.shape, pred_disp_SGBM.shape))
    print("'''''''''''''''''''''''SGBM Evaluation Result''''''''''''''''''''''''''''''")
    evaluate_disparity(gt_disparity, pred_disp_SGBM)
    print("CRE: gt and pred shape: {0}, {1}".format(gt_disparity.shape, predicted_disparity_CRE.shape))
    print("'''''''''''''''''''''''SGBM Evaluation Result''''''''''''''''''''''''''''''")
    evaluate_disparity(gt_disparity, predicted_disparity_CRE)

    compare = np.hstack([pred_vis_SGBM, gt_vis, pred_vis_CRE])
    cv2.imshow("compare", compare)
    cv2.waitKey()
    cv2.destroyAllWindows()
    cv2.imwrite("../data/cones-png-2/cones/disp_cre.png", predicted_disparity_CRE)
