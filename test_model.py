import pandas as pd
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from imread_from_url import imread_from_url
from disparity_depth import compute_utils
from nets import Model
import pydeck
import open3d as o3d

device = 'cuda'

#Ref: https://github.com/megvii-research/CREStereo/blob/master/test.py
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

if __name__ == '__main__':

	# left_img = imread_from_url("https://raw.githubusercontent.com/megvii-research/CREStereo/master/img/test/left.png")
	# right_img = imread_from_url("https://raw.githubusercontent.com/megvii-research/CREStereo/master/img/test/right.png")
	left_img = cv2.imread("disparity_depth/image/kochi/left.png")
	right_img = cv2.imread("disparity_depth/image/kochi/right.png")

	in_h, in_w = left_img.shape[:2]

	# Resize image in case the GPU memory overflows
	eval_h, eval_w = (in_h,in_w)
	assert eval_h%8 == 0, "input height should be divisible by 8"
	assert eval_w%8 == 0, "input width should be divisible by 8"
	
	imgL = cv2.resize(left_img, (eval_w, eval_h), interpolation=cv2.INTER_LINEAR)
	imgR = cv2.resize(right_img, (eval_w, eval_h), interpolation=cv2.INTER_LINEAR)

	model_path = "models/crestereo_eth3d.pth"

	model = Model(max_disp=256, mixed_precision=False, test_mode=True)
	model.load_state_dict(torch.load(model_path), strict=True)
	model.to(device)
	model.eval()

	pred = inference(imgL, imgR, model, n_iter=20)
	t = float(in_w) / float(eval_w)
	disp = cv2.resize(pred, (in_w, in_h), interpolation=cv2.INTER_LINEAR) * t

	fx = 260.27365
	baseline = 119.84703
	depth = compute_utils.compute_depth(disp, baseline, fx)

	width = left_img.shape[1]
	height = left_img.shape[0]
	fx = 260.05169
	fy = 260.05169
	cx = 335.19976
	cy = 191.58966
	camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

	color_raw = o3d.geometry.Image(left_img)
	depth_raw = o3d.geometry.Image(depth)
	print("Color, depth types: {0}, {1}".format(color_raw.Type, depth_raw.Type))
	rgbd_img = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw, convert_rgb_to_intensity=False)
	pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_img, camera_intrinsic)
	pcd.transform([
		[1, 0, 0, 0],
		[0, -1, 0, 0],
		[0, 0, -1, 0],
		[0, 0, 0, 1]
	])
	# o3d.visualization.draw_geometries([pcd])

	o3d.io.write_point_cloud("./test_data/test_point_cloud.pcd", pcd, print_progress=True)
	# pcd_points = np.asarray(pcd.points)
	# pcd_colors = np.asarray(pcd.colors)
	# # print(pcd_points[:5, :])
	# # print(pcd_colors[:5, :])
	# pcd_rgb = np.concatenate((pcd_points, pcd_colors), axis=1)
	# # print(pcd_rgb[:5, :])
	# print(pcd_rgb.shape)
	# df = pd.DataFrame(pcd_rgb, columns=['x', 'y', 'z', 'r', 'g', 'b'])
	# u, v = np.meshgrid(np.arange(width), np.arange(height))
	# z = depth / 1000
	# x = (u - cx) * z / fx
	# y = (v - cy) * z / fy
	# # z = depth
	# pcd = np.stack((x, y, z), axis=-1)
	# # pcd_o3d = compute_utils.depth2pcd_with_o3d(depth, camera_intrinsic)
	# # print("Pcd have colors: ", pcd_o3d.has_colors())
	# # pcd_points = np.asarray(pcd_o3d.points)
	# print("Pcd shape:" , pcd.shape)
	# print("Left image shape: ", left_img.shape)
	# pcd_with_rgb = np.concatenate((pcd, left_img), axis=2)
	#
	# pcd_flat = pcd_with_rgb.reshape(-1, 6)
	# df = pd.DataFrame(pcd_flat, columns=['x', 'y', 'z', 'r', 'g', 'b'])
	# # df.reset_index(drop=True, inplace=True)
	#
	# visualization
	# disp_vis = (disp - disp.min()) / (disp.max() - disp.min()) * 255.0
	# disp_vis = disp_vis.astype("uint8")
	# disp_vis = cv2.applyColorMap(disp_vis, cv2.COLORMAP_INFERNO)
	#
	# depth_vis = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
	# depth_vis = depth_vis.astype("uint8")
	# depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)
	#
	# target = [df.x.mean(), df.y.mean(), df.z.mean()]
	# pcd_layer = pydeck.Layer(
	# 	"PointCloudLayer",
	# 	data=df,
	# 	get_position=["x", "y", "z"],
	# 	auto_highlight=True,
	# 	pickable=True,
	# 	point_size=3
	# )
	#
	# df.to_csv("kochi_pcd.csv", index=False)
	#
	# view_state = pydeck.ViewState(target=target, controller=True, rotation_x=15, rotation_orbit=30, zoom=5.3)
	# view = pydeck.View(type="OrbitView", controller=True)
	#
	# r = pydeck.Deck(pcd_layer, initial_view_state=view_state, views=[view])
	# r.to_html("point_cloud_layer.html", css_background_color="#add8e6")

	# combined_img = np.hstack((left_img, disp_vis))
	# cv2.namedWindow("output", cv2.WINDOW_NORMAL)
	# cv2.imshow("left and right", combined_img)
	# disp_depth = np.hstack([disp_vis, depth_vis])
	# cv2.imshow("disp and depth", disp_depth)
	# # cv2.imwrite("output.jpg", disp_vis)
	# cv2.waitKey(0)



