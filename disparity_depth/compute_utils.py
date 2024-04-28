import time
import torch
import torch.nn.functional as F
import cv2
import numpy as np
import open3d as o3d
from nets import Model

device = 'cuda'
# model_path = "models/crestereo_eth3d.pth"
# model_path = "../models/crestereo_eth3d.pth"
model_path = "E:\grade_4\graduate\ZED\models\crestereo_eth3d.pth"
model = Model(max_disp=256, mixed_precision=False, test_mode=True)
model.load_state_dict(torch.load(model_path), strict=True)
model.to(device)
model.eval()

def compute_disparity_SGBM(left, right,
                           min_disp, num_disp,
                           block_size,
                           P1, P2,
                           disp12MaxDiff,
                           preFilterCap,
                           uniquenessRatio,
                           speckleWindowSize, speckleRange,
                           mode):
    # 创建一个StereoSGBM实例
    stereo = cv2.StereoSGBM.create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=block_size,
        P1=P1,
        P2=P2,
        disp12MaxDiff=disp12MaxDiff,
        preFilterCap=preFilterCap,
        uniquenessRatio=uniquenessRatio,
        speckleWindowSize=speckleWindowSize,
        speckleRange=speckleRange,
        mode=mode
    )

    # openCV returns 16 * disparity in pixels
    disp = stereo.compute(left, right).astype(np.float32) / 16.0
    # disp = stereo.compute(left, right).astype(np.float32)
    # 视差图归一化到 [0, 1]的范围内
    # disp = np.clip((disp - min_disp) / num_disp, 0, 1)

    return disp


def inference(left, right, model, n_iter=20):
    print("Model Forwarding...")
    imgL = left.transpose(2, 0, 1)
    imgR = right.transpose(2, 0, 1)
    imgL = np.ascontiguousarray(imgL[None, :, :, :])
    imgR = np.ascontiguousarray(imgR[None, :, :, :])

    imgL = torch.tensor(imgL.astype("float32")).to(device)
    imgR = torch.tensor(imgR.astype("float32")).to(device)

    imgL_dw2 = F.interpolate(
        imgL,
        size=(imgL.shape[2] // 2, imgL.shape[3] // 2),
        mode="bilinear",
        align_corners=True,
    )
    imgR_dw2 = F.interpolate(
        imgR,
        size=(imgL.shape[2] // 2, imgL.shape[3] // 2),
        mode="bilinear",
        align_corners=True,
    )
    # print(imgR_dw2.shape)
    with torch.inference_mode():
        pred_flow_dw2 = model(imgL_dw2, imgR_dw2, iters=n_iter, flow_init=None)

        pred_flow = model(imgL, imgR, iters=n_iter, flow_init=pred_flow_dw2)
    pred_disp = torch.squeeze(pred_flow[:, 0, :, :]).cpu().detach().numpy()

    return pred_disp


def compute_disparity_CRE(left_img, right_img):
    # BGRA convert to BGR
    left_img = cv2.cvtColor(left_img, cv2.COLOR_BGRA2BGR)
    right_img = cv2.cvtColor(right_img, cv2.COLOR_BGRA2BGR)

    in_h, in_w = left_img.shape[:2]

    # Resize image in case the GPU memory overflows
    eval_h, eval_w = (in_h, in_w)
    assert eval_h % 8 == 0, "input height should be divisible by 8"
    assert eval_w % 8 == 0, "input width should be divisible by 8"

    imgL = cv2.resize(left_img, (eval_w, eval_h), interpolation=cv2.INTER_LINEAR)
    imgR = cv2.resize(right_img, (eval_w, eval_h), interpolation=cv2.INTER_LINEAR)

    # # model_path = "models/crestereo_eth3d.pth"
    # model_path = "../models/crestereo_eth3d.pth"
    #
    # model = Model(max_disp=256, mixed_precision=False, test_mode=True)
    # model.load_state_dict(torch.load(model_path), strict=True)
    # model.to(device)
    # model.eval()

    pred = inference(imgL, imgR, model, n_iter=20)

    t = float(in_w) / float(eval_w)
    disp = cv2.resize(pred, (in_w, in_h), interpolation=cv2.INTER_LINEAR) * t

    # disp_vis = (disp - disp.min()) / (disp.max() - disp.min()) * 255.0
    # disp_vis = disp_vis.astype("uint8")
    # disp_vis = cv2.applyColorMap(disp_vis, cv2.COLORMAP_INFERNO)
    #
    # combined_img = np.hstack((left_img, disp_vis))
    # cv2.namedWindow("output", cv2.WINDOW_NORMAL)
    # cv2.imshow("output", combined_img)
    # cv2.imwrite("output.jpg", disp_vis)
    # cv2.waitKey(0)

    return disp


def img_visualize(img):
    # visualization
    img_vis = (img - img.min()) / (img.max() - img.min()) * 255.0
    img_vis = img_vis.astype("uint8")
    img_vis = cv2.applyColorMap(img_vis, cv2.COLORMAP_INFERNO)
    return img_vis


# 深度图
def compute_depth(disparity_map, baseline, fx):
    """
    Compute depth map from disparity map
    :param disparity_map:
    :param baseline: baseline of camera (unit: mm)
    :param fx: focal length of camera
    :return: depth map
    """
    # Depth = 焦距 * 基线距离 / 视差
    depth_map = (fx * baseline) / disparity_map
    return depth_map


def depth2pcd(depth_map, camera_intrinsics):
    # todo: 加入上色代码
    width = camera_intrinsics.width
    height = camera_intrinsics.height
    intrinsic_matrix = camera_intrinsics.intrinsic_matrix
    fx = intrinsic_matrix[0][0]
    fy = intrinsic_matrix[1][1]
    cx = intrinsic_matrix[0][2]
    cy = intrinsic_matrix[1][2]
    # width, height, fx, fy, cx, cy = camera_intrinsics
    # Generate pixel grid
    u, v = np.meshgrid(np.arange(width), np.arange(height))

    # compute x, y, z coordinates
    x = (u - cx) * depth_map / fx
    y = (v - cy) * depth_map / fy
    z = depth_map

    pcd = np.dstack((x, y, z))

    return pcd


def depth2pcd_with_o3d(left_img, depth_map, intrinsic):
    color_raw = o3d.geometry.Image(left_img)
    depth_raw = o3d.geometry.Image(depth_map)
    rgbd_img = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw, convert_rgb_to_intensity=False)
    # print(depth_image.type, intrinsic.type)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_img, intrinsic)
    # Flip point cloud, otherwise it will be upside down
    pcd.transform([
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1]
    ])
    return pcd


def visualize_pcd(point_cloud):
    # pcd = o3d.io.read_point_cloud(point_cloud)
    o3d.visualization.draw([point_cloud])
    # o3d.visualization.draw_geometries([point_cloud])


def length_to_pixels(length_mm, sensor_width_mm, image_width):
    """
    Convert length from millimeters to pixels.

    Parameters:
    :param length_mm: Length in millimeters.
    :param sensor_width_mm: Sensor width in millimeters.
    :param image_width: Image width in pixels

    Returns:
    :return pixels: Length in pixels
    """
    pixels = length_mm / (sensor_width_mm / image_width)
    return pixels


def write_ply(point_cloud, save_ply):
    start = time.time()
    float_formatter = lambda x: "%.4f" % x
    points = []

    for i in point_cloud.T:
        points.append("{} {} {} {} {} {} o\n".format(
            float_formatter(i[0]), float_formatter(i[1]), float_formatter(i[2]),
            int(i[3]), int(i[4]), int(i[5])
        ))

    file = open(save_ply, "w")
    file.write(
        '''ply
        format ascii 1.0
        element vertex %d
        property float x
        property float y
        property float z
        property uchar red
        property uchar green
        property uchar blue
        property uchar alpha
        end_header
        %s
        ''' % (len(points), "".join(points))
    )
    file.close()

    end = time.time()
    print("Write into .ply file Done", end - start)
