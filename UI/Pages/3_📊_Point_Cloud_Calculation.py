import numpy as np
import streamlit as st
from PIL import Image
from matplotlib import pyplot as plt
import open3d as o3d

from disparity_depth import compute_utils

st.set_page_config(page_title="Point Cloud Calculation", page_icon=":smile:")

st.title("Point Cloud Calculator")

# Upload depth map
depth_map = st.file_uploader("Upload Depth Map", type=['jpg', 'png'])

if depth_map is not None:
    depth_map = np.array(Image.open(depth_map))
    st.image(depth_map, caption="Uploaded Depth Map", use_column_width=True)
    width = depth_map.shape[1]
    height = depth_map.shape[0]

    # Sidebar for adjusting camera intrinsics
    st.sidebar.title("Camera Intrinsics")
    st.sidebar.write("Width: ", width)
    st.sidebar.write("Height: ", height)
    fx = st.sidebar.number_input("fx", min_value=1, max_value=1000, value=100)
    fy = st.sidebar.number_input("fy", min_value=1, max_value=1000, value=100)
    cx = st.sidebar.number_input("cx", min_value=0, max_value=depth_map.shape[1] - 1, value=depth_map.shape[1] // 2)
    cy = st.sidebar.number_input("cy", min_value=0, max_value=depth_map.shape[0] - 1, value=depth_map.shape[0] // 2)

    camera_intrinsics = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

    # Compute point cloud
    point_cloud = compute_utils.depth2pcd(depth_map, camera_intrinsics)

    # Display point cloud
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(point_cloud[:, :, 0], point_cloud[:, :, 1], point_cloud[:, :, 2], c=depth_map.flatten(), cmap='gray')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    st.pyplot(fig)

    # Save point cloud
    if st.button("Save Point Cloud"):
        o3d.io.write_point_cloud("point_cloud.ply", point_cloud)
        st.success("Point Cloud Saved Successfully!")
