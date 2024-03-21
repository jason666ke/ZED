import numpy as np
import streamlit as st
from PIL import Image

import get_image


def compute_disp_ui():
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
    min_disp = st.sidebar.slider("Minimum Disparity", min_value=0, max_value=16, value=0)
    num_disp = st.sidebar.slider("Number of Disparities", min_value=16, max_value=256, value=16, step=16)
    block_size = st.sidebar.slider("Block Size", min_value=3, max_value=13, value=3, step=2)
    uniquenessRatio = st.sidebar.slider("Uniqueness Ratio", min_value=10, max_value=50, value=10)
    speckleRange = st.sidebar.slider("Speckle Range", min_value=0, max_value=20, value=0)
    speckleWindowSize = st.sidebar.slider("Speckle Window Size", min_value=0, max_value=20, value=0)
    disp12MaxDiff = st.sidebar.slider("disp12MaxDiff", min_value=-1, max_value=50, value=-1)
    P1 = st.sidebar.slider("P1", min_value=8 * block_size, max_value=16 * block_size, value=8 * block_size)
    P2 = st.sidebar.slider("P2", min_value=32 * block_size, max_value=256 * block_size, value=32 * block_size)

    # button to start computation
    compute_button = st.sidebar.button("Compute Disparity")

    if compute_button:
        if left_img and right_img:
            left_img = Image.open(left_img)
            right_img = Image.open(right_img)

            # convert numpy arrays to UMat objects
            left_img = np.array(left_img)
            right_img = np.array(right_img)

            # compute disparity map
            disparity_map = get_image.compute_disparity(left_img, right_img, min_disp, num_disp, block_size,
                                                        uniquenessRatio, speckleRange, speckleWindowSize, disp12MaxDiff,
                                                        P1, P2)

            # Display disparity map
            if disparity_map is not None:
                st.image(disparity_map, caption="Disparity Map", use_column_width=True)
            else:
                st.error("Failed to compute disparity map.")
        else:
            st.error("Please upload left and right image before computing disparity.")


if __name__ == '__main__':
    compute_disp_ui()
