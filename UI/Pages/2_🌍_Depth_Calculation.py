import cv2
import numpy as np
import streamlit as st
from PIL import Image
from disparity_depth import compute_utils

st.set_page_config(page_title="Depth Calculation", page_icon="🌍")

st.title("Depth Calculation")

# upload disparity map
disparity_img = st.sidebar.file_uploader("Upload Disparity", type=['jpg', 'jpeg', 'png'])
if disparity_img:
    disparity_img = Image.open(disparity_img)
    st.image(disparity_img, caption="Uploaded Disparity Image", use_column_width=True)

# sidebar for adjusting parameters
st.sidebar.header("Adjust Parameters")
baseline_mm = st.sidebar.number_input("Baseline (mm)", min_value=1, max_value=1000, value=100)
focal_length_px = st.sidebar.number_input("Focal Length (pixels)", min_value=1, max_value=1000, value=100)

# compute depth map
if disparity_img:
    depth_map = compute_utils.compute_depth(np.array(disparity_img), baseline_mm, focal_length_px)
    # display depth map
    depth_map_0_1 = cv2.normalize(depth_map, None, 0, 1, cv2.NORM_MINMAX)
    st.image(depth_map_0_1, caption="Depth", use_column_width=True)
else:
    st.error("Please upload disparity image before computing depth.")





