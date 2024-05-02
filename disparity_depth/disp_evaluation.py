import numpy as np
import cv2
import compute_utils


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
    ssim = cv2.SSIM(np.float32(gt_disparity), np.float32(predicted_disparity))
    return ssim


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
    mae = compute_mae(gt_disparity, predicted_disparity)
    epe = compute_epe(gt_disparity, predicted_disparity)
    # ssim = compute_ssim(gt_disparity, predicted_disparity)

    # 打印结果
    print("Mean Squared Error (MSE):", mse)
    print("Mean Absolute Error (MAE):", mae)
    print("End-point Error (EPE):", np.mean(epe))
    # print("Structural Similarity Index (SSIM):", ssim)


if __name__ == "__main__":
    left = cv2.imread("../data/cones-png-2/cones/im2.png")
    right = cv2.imread("../data/cones-png-2/cones/im6.png")

    print("left, right shape: {0}, {1}".format(left.shape, right.shape))
    model = compute_utils.load_model()
    # 输入真实视差图和预测视差图的路径
    gt_disparity_path = "../data/cones-png-2/cones/disp2.png"
    gt_disparity = get_disp(gt_disparity_path)
    gt_vis = compute_utils.img_visualize(gt_disparity)
    predicted_disparity = compute_utils.compute_disparity_CRE(left, right, model) / 8
    pred_vis = compute_utils.img_visualize(predicted_disparity)
    print("gt and pred shape: {0}, {1}".format(gt_disparity.shape, predicted_disparity.shape))

    # 评估视差图
    evaluate_disparity(gt_disparity, predicted_disparity)

    compare = np.hstack([gt_vis, pred_vis])
    cv2.imshow("compare", compare)
    cv2.waitKey()
    cv2.destroyAllWindows()
    cv2.imwrite("../data/cones-png-2/cones/disp_cre.png", predicted_disparity)
