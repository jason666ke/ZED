import datetime
import io
import zipfile

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from disparity_depth import compute_utils


@st.cache_resource
def load_model():
    return compute_utils.load_model()


st.set_page_config(page_title="Disparity Calculation", page_icon="📈")

st.title("Disparity Map Calculation")

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
P1 = st.sidebar.slider("P1", min_value=8 * 3 * block_size, max_value=16 * 3 * block_size, value=8 * 3 * block_size,
                       key="P1")
P2 = st.sidebar.slider("P2", min_value=32 * 3 * block_size, max_value=256 * 3 * block_size, value=32 * 3 * block_size,
                       key="P2")
disp12MaxDiff = st.sidebar.slider("disp12MaxDiff", min_value=-1, max_value=50, value=-1, key="disp12MaxDiff")
preFilterCap = st.sidebar.slider("preFilterCap", min_value=1, max_value=63, value=30, key="preFilterCap")
uniquenessRatio = st.sidebar.slider("Uniqueness Ratio", min_value=10, max_value=50, value=10, key="uniquenessRatio")
speckleWindowSize = st.sidebar.slider("Speckle Window Size", min_value=0, max_value=20, value=0,
                                      key="speckleWindowSize")
speckleRange = st.sidebar.slider("Speckle Range", min_value=0, max_value=20, value=0, key="speckleRange")

# SGBM mode select
mode_functions = {
    "SGBM": cv2.STEREO_SGBM_MODE_SGBM,
    "HH": cv2.STEREO_SGBM_MODE_HH,
    "SGBM_3WAY": cv2.STEREO_SGBM_MODE_SGBM_3WAY
}
mode_button = st.sidebar.radio("Select Mode", ("SGBM", "HH", "SGBM_3WAY"))
select_mode = mode_functions.get(mode_button)

model = load_model()
# compute disparity after upload left and right image
if left_img and right_img:
    left_img = Image.open(left_img)
    right_img = Image.open(right_img)

    # convert numpy arrays to UMat objects
    left_img = np.array(left_img)
    right_img = np.array(right_img)

    # compute disparity map
    disparity_SGBM = compute_utils.compute_disparity_SGBM(left_img, right_img, min_disp, num_disp, block_size,
                                                          P1, P2, disp12MaxDiff, preFilterCap,
                                                          uniquenessRatio, speckleWindowSize,
                                                          speckleRange, select_mode)
    disparity_CRE = compute_utils.compute_disparity_CRE(left_img, right_img, model)

    if disparity_SGBM is not None:
        # Display disparity map
        # disparity_map_0_1 = cv2.normalize(disparity_SGBM, None, 0, 1, cv2.NORM_MINMAX)
        disp_vis_SGBM = compute_utils.img_visualize(disparity_SGBM)
        disp_vis_CRE = compute_utils.img_visualize(disparity_CRE)
        with col1:
            st.image(disp_vis_SGBM, caption="SGBM", use_column_width=True, clamp=True)
        with col2:
            st.image(disp_vis_CRE, caption="CRE", use_column_width=True, clamp=True)
        # st.image(disparity_map_0_1, caption="Disparity Map", use_column_width=True)
        # download disparity
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'a', zipfile.ZIP_DEFLATED) as zipfile:
            with zipfile.open("SGBM_disparity.png", 'w') as f:
                f.write(cv2.imencode('.png', disp_vis_SGBM)[1].tobytes())
            with zipfile.open("CRE_disparity.png", 'w') as f:
                f.write(cv2.imencode('.png', disp_vis_CRE)[1].tobytes())
        zip_buffer.seek(0)
        st.download_button("Download Disparity Maps", zip_buffer.getvalue(),
                           file_name="disparity_maps.zip",
                           mime="application/zip")
    else:
        st.error("Failed to compute disparity map.")
else:
    st.info("Please upload left and right image before computing disparity.")
