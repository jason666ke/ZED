import streamlit as st
import pyzed.sl as sl
from disparity_depth import camera_utils
import disparity_depth.camera_utils

st.set_page_config(page_title="Real-time Estimation", page_icon="📹")

st.title("Real-time Estimation")

# sidebar for setting camera parameters
st.sidebar.title("Camera Initialization Parameters")
resolution_options = [sl.RESOLUTION.AUTO, sl.RESOLUTION.VGA,
                      sl.RESOLUTION.HD720, sl.RESOLUTION.HD1200]
depth_mode_options = [sl.DEPTH_MODE.ULTRA, sl.DEPTH_MODE.NEURAL,
                      sl.DEPTH_MODE.LAST, sl.DEPTH_MODE.PERFORMANCE, sl.DEPTH_MODE.QUALITY]

selected_resolution = st.sidebar.selectbox("Resolution", resolution_options, format_func=lambda x: x.name)
selected_depth_mode = st.sidebar.selectbox("Depth Mode", depth_mode_options, format_func=lambda x: x.name)

# connect flag
connect_flag = False

zed = sl.Camera()
connect_button = st.sidebar.button("Connect")
if connect_button:
    init = sl.InitParameters(depth_mode=selected_depth_mode, camera_resolution=selected_resolution)
    err = zed.open(init)
    if err != sl.ERROR_CODE.SUCCESS:
        st.error("Failed to connect to ZED camera. Error code: {}".format(err))
        connect_flag = False
    else:
        # print camera information
        camera_info = zed.get_camera_information()
        st.write("Camera Info:", camera_info)
        connect_flag = True

runtime_params = sl.RuntimeParameters()
stop_button = st.sidebar.button("Stop")

# left and right image from camera
left_img = sl.Mat()
right_img = sl.Mat()
# left and right image container
left_container = st.empty()
right_container = st.empty()
col1, col2 = st.columns(2)

while connect_flag:
    if stop_button:
        connect_flag = False
    # capture frame
    if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
        # retrieve left image and right image
        zed.retrieve_image(left_img, sl.VIEW.LEFT)
        zed.retrieve_image(right_img, sl.VIEW.RIGHT)
        # display left and right image
        left_data = left_img.get_data()
        right_data = right_img.get_data()
        with col1:
            left_container.image(left_data, caption="Left Image", use_column_width=True)
        with col2:
            right_container.image(right_data, caption="Right Image", use_column_width=True)

