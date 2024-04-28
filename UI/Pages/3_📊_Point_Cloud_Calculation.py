import numpy as np
import pandas as pd
import pydeck
import pydeck as pdk
import streamlit as st
from PIL import Image
from matplotlib import pyplot as plt
import open3d as o3d
import streamlit.components.v1 as components

from disparity_depth import compute_utils

st.set_page_config(page_title="Point Cloud Calculation", page_icon=":smile:")

st.title("Point Cloud Calculator")

# upload left image
left_img = st.sidebar.file_uploader("Upload Left Image", type=['jpg', 'png', 'jpeg'])
# Upload right map
right_img = st.sidebar.file_uploader("Upload Right Map", type=['jpg', 'png', 'jpeg'])

col_1, col_2 = st.columns(2)
with col_1:
    if left_img:
        st.image(left_img, caption="Left image", use_column_width=True, clamp=True)
with col_2:
    if right_img:
        st.image(right_img, caption="Right image", use_column_width=True, clamp=True)

# sidebar for disparity and depth
st.sidebar.title("Depth parameters")
baseline_mm = st.sidebar.number_input("Baseline (mm)", min_value=1, max_value=1000, value=120)
focal_length_px = st.sidebar.number_input("Focal Length (pixels)", min_value=1, max_value=1000, value=260)

if right_img and left_img is not None:
    left_img = Image.open(left_img)
    right_img = Image.open(right_img)
    # convert to numpy arrays
    left_img = np.array(left_img)
    right_img = np.array(right_img)
    # display left image size
    width = left_img.shape[1]
    height = left_img.shape[0]
    # Sidebar for adjusting camera intrinsics
    st.sidebar.title("Camera Intrinsics")
    st.sidebar.write("Width: ", width)
    st.sidebar.write("Height: ", height)
    fx = st.sidebar.number_input("fx", min_value=1, max_value=1000, value=100)
    fy = st.sidebar.number_input("fy", min_value=1, max_value=1000, value=100)
    cx = st.sidebar.number_input("cx", min_value=0, max_value=right_img.shape[1] - 1, value=right_img.shape[1] // 2)
    cy = st.sidebar.number_input("cy", min_value=0, max_value=right_img.shape[0] - 1, value=right_img.shape[0] // 2)

    # compute disparity and depth
    disp = compute_utils.compute_disparity_CRE(left_img, right_img)
    depth = compute_utils.compute_depth(disp, baseline_mm, focal_length_px)
    # visualize disparity and depth
    with col_1:
        disp_vis = compute_utils.img_visualize(disp)
        st.image(disp_vis, caption="Disparity", use_column_width=True, clamp=True)
    with col_2:
        depth_vis = compute_utils.img_visualize(depth)
        st.image(depth_vis, caption="Depth", use_column_width=True, clamp=True)

    camera_intrinsics = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

    # Compute point cloud
    point_cloud = compute_utils.depth2pcd_with_o3d(left_img, depth, camera_intrinsics)
    point_cloud.estimate_normals()

    # Display point cloud
    points = np.asarray(point_cloud.points)
    colors = np.asarray(point_cloud.colors)
    normals = np.asarray(point_cloud.normals)

    # raw data
    pcd_df = pd.DataFrame({
        'x': points[:, 0],
        'y': points[:, 1],
        'z': points[:, 2],
        'normalX': normals[:, 0],
        'normalY': normals[:, 1],
        'normalZ': normals[:, 2],
        'colorR': colors[:, 0],
        'colorG': colors[:, 1],
        'colorB': colors[:, 2]
    })
    # pcd Layer
    pcd_layer = pydeck.Layer(
        "PointCloudLayer",
        data=pcd_df,
        get_position=["x", "y", "z"],
        get_color=["colorR", "colorG", "colorB"],
        get_normal=["normalX", "normalY", "normalZ"],
        pickable=True,
        auto_highlight=True
    )
    # view state
    target = points.mean()
    view_state = pydeck.ViewState(target=target, controller=True, rotation_orbit=30, zoom=5.3)
    view = pydeck.View(type="OrbitView", controller=True)
    # deck
    deck = pdk.Deck(
        layers=[pcd_layer],
        initial_view_state=view_state,
        views=[view]
    )

    st.pydeck_chart(deck)

    # Save point cloud
    if st.button("Save Point Cloud"):
        o3d.io.write_point_cloud("point_cloud.ply", point_cloud)
        st.success("Point Cloud Saved Successfully!")


