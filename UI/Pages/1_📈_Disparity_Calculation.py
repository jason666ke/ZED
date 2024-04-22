import datetime
import io

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from disparity_depth import compute_utils

st.set_page_config(page_title="Disparity Calculation", page_icon="📈")

st.title("Disparity Map Calculation")

# back_button = st.sidebar.button("Return to Home Page")
# # if back_button:

# upload left and right images
st.sidebar.header("Upload Images")
left_img = st.sidebar.file_uploader("Upload Left Image", type=['jpg', 'jpeg', 'png'])
right_img = st.sidebar.file_uploader("Upload Right Image", type=['jpg', 'jpeg', 'png'])
# display uploaded images
col1, col2 = st.columns(2)
with col1:
    if left_img:
        st.header("Left Image")
        st.image(left_img, caption="Left Image", use_column_width=True)
with col2:
    if right_img:
        st.header("Right Image")
        st.image(right_img, caption="Right Image", use_column_width=True)

# parameters for StereoSGBM
st.sidebar.header("Adjust Parameters")
min_disp = st.sidebar.slider("Minimum Disparity", min_value=0, max_value=16, value=0, key="min_disp")
num_disp = st.sidebar.slider("Number of Disparities", min_value=16, max_value=256, value=16, step=16, key="num_disp")
block_size = st.sidebar.slider("Block Size", min_value=3, max_value=13, value=3, step=2, key="block_size")
P1 = st.sidebar.slider("P1", min_value=8 * 3 * block_size, max_value=16 * 3 * block_size, value=8 * 3 * block_size, key="P1")
P2 = st.sidebar.slider("P2", min_value=32 * 3 * block_size, max_value=256 * 3 * block_size, value=32 * 3 * block_size, key="P2")
disp12MaxDiff = st.sidebar.slider("disp12MaxDiff", min_value=-1, max_value=50, value=-1, key="disp12MaxDiff")
preFilterCap = st.sidebar.slider("preFilterCap", min_value=1, max_value=63, value=30, key="preFilterCap")
uniquenessRatio = st.sidebar.slider("Uniqueness Ratio", min_value=10, max_value=50, value=10, key="uniquenessRatio")
speckleWindowSize = st.sidebar.slider("Speckle Window Size", min_value=0, max_value=20, value=0, key="speckleWindowSize")
speckleRange = st.sidebar.slider("Speckle Range", min_value=0, max_value=20, value=0, key="speckleRange")

# SGBM mode select
mode_functions = {
    "SGBM": cv2.STEREO_SGBM_MODE_SGBM,
    "HH": cv2.STEREO_SGBM_MODE_HH,
    "SGBM_3WAY": cv2.STEREO_SGBM_MODE_SGBM_3WAY
}
mode_button = st.sidebar.radio("Select Mode", ("SGBM", "HH", "SGBM_3WAY"))
select_mode = mode_functions.get(mode_button)

# compute disparity after upload left and right image
if left_img and right_img:
    left_img = Image.open(left_img)
    right_img = Image.open(right_img)

    # convert numpy arrays to UMat objects
    left_img = np.array(left_img)
    right_img = np.array(right_img)

    # compute disparity map
    disparity_map = compute_utils.compute_disparity(left_img, right_img, min_disp, num_disp, block_size,
                                                    P1, P2, disp12MaxDiff, preFilterCap,
                                                    uniquenessRatio, speckleWindowSize,
                                                    speckleRange, select_mode)

    if disparity_map is not None:
        # Display disparity map
        disparity_map_0_1 = cv2.normalize(disparity_map, None, 0, 1, cv2.NORM_MINMAX)
        st.image(disparity_map_0_1, caption="Disparity Map", use_column_width=True)

        # convert disparity map to image
        disp_img = Image.fromarray(disparity_map.astype(np.uint8))

        # button to save disparity
        buffer = io.BytesIO()
        disp_img.save(buffer, format='PNG')
        buffer.seek(0)

        # get file name
        cur_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        file_name = "disparity_map" + cur_time + ".png"

        disp_download_button = st.sidebar.download_button(label="Save Disparity Map", data=buffer,
                                                          file_name=file_name, mime="image/png")
    else:
        st.error("Failed to compute disparity map.")
else:
    st.error("Please upload left and right image before computing disparity.")



