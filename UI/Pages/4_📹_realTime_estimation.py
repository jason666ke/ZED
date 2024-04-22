import cv2
import streamlit as st
import pyzed.sl as sl
from disparity_depth import camera_utils
from disparity_depth import compute_utils

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
connect_button = st.sidebar.button("Connect")
stop_button = st.sidebar.button("Stop")

# parameters for StereoSGBM
st.sidebar.header("Adjust Parameters")
min_disp = st.sidebar.slider("Minimum Disparity", min_value=0, max_value=16, value=0, key="min_disp")
num_disp = st.sidebar.slider("Number of Disparities", min_value=16, max_value=256, value=96, step=16, key="num_disp")
block_size = st.sidebar.slider("Block Size", min_value=3, max_value=13, value=7, step=2, key="block_size")
P1 = st.sidebar.slider("P1", min_value=8 * 3 * block_size, max_value=16 * 3 * block_size, value=8 * 3 * block_size, key="P1")
P2 = st.sidebar.slider("P2", min_value=32 * 3 * block_size, max_value=256 * 3 * block_size, value=32 * 3 * block_size, key="P2")
disp12MaxDiff = st.sidebar.slider("disp12MaxDiff", min_value=-1, max_value=50, value=1, key="disp12MaxDiff")
preFilterCap = st.sidebar.slider("preFilterCap", min_value=1, max_value=63, value=63, key="preFilterCap")
uniquenessRatio = st.sidebar.slider("Uniqueness Ratio", min_value=10, max_value=50, value=10, key="uniquenessRatio")
speckleWindowSize = st.sidebar.slider("Speckle Window Size", min_value=0, max_value=200, value=100, key="speckleWindowSize")
speckleRange = st.sidebar.slider("Speckle Range", min_value=0, max_value=20, value=10, key="speckleRange")

# SGBM mode select
mode_functions = {
    "SGBM": cv2.STEREO_SGBM_MODE_SGBM,
    "HH": cv2.STEREO_SGBM_MODE_HH,
    "SGBM_3WAY": cv2.STEREO_SGBM_MODE_SGBM_3WAY
}
mode_button = st.sidebar.radio("Select Mode", ("SGBM", "HH", "SGBM_3WAY"))
select_mode = mode_functions.get(mode_button)

# connect flag
connect_flag = False

zed = sl.Camera()
init = sl.InitParameters(depth_mode=selected_depth_mode, camera_resolution=selected_resolution)


if connect_button:
    err = zed.open(init)
    if err != sl.ERROR_CODE.SUCCESS:
        st.error("Failed to connect to ZED camera. Error code: {}".format(err))
    else:
        # camera information
        camera_info = zed.get_camera_information()
        input_type = camera_info.input_type
        model = camera_info.camera_model
        serial_number = camera_info.serial_number
        # camera config
        camera_config = camera_info.camera_configuration
        # resolution = camera_config.resolution
        width = camera_config.resolution.width
        height = camera_config.resolution.height
        baseline = camera_config.calibration_parameters.get_camera_baseline()
        fps = camera_config.fps
        # left cam parameters
        left_cam = camera_config.calibration_parameters.left_cam
        fx = left_cam.fx
        fy = left_cam.fy
        cx = left_cam.cx
        cy = left_cam.cy
        info_col, config_col, cam_col = st.columns(3)
        with info_col:
            info_col.caption("Camera Information")
            info_col.write("Input type: {}".format(input_type))
            info_col.write("Camera model: {}".format(model))
            info_col.write("Selected Resolution: {}".format(selected_resolution))
            info_col.write("Serial number: {}".format(serial_number))
        with config_col:
            config_col.caption("Camera Configuration")
            # config_col.write("Resolution: {}".format(resolution))
            config_col.write("Width: {} pixels".format(width))
            config_col.write("Height: {} pixels".format(height))
            config_col.write("Baseline: {:.5f} millimeters".format(baseline))
            config_col.write("Fps: {}".format(fps))
        with cam_col:
            cam_col.caption("Camera Parameters")
            cam_col.write("fx: {:.5f} pixels".format(fx))
            cam_col.write("fy: {:.5f} pixels".format(fy))
            cam_col.write("cx: {:.5f} pixels".format(cx))
            cam_col.write("cy: {:.5f} pixels".format(cy))

        # st.write("Camera Info:", camera_info)
        connect_flag = True

runtime_params = sl.RuntimeParameters()


# left and right image from camera
left_img = sl.Mat()
right_img = sl.Mat()
# left and right image container
left_container = st.empty()
right_container = st.empty()
# disparity container
disp_container = st.empty()
# depth container
depth_container = st.empty()

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
        col1, col2 = st.columns(2)
        with col1:
            left_container.image(left_data, caption="Left Image", use_column_width=True)
        with col2:
            right_container.image(right_data, caption="Right Image", use_column_width=True)

        # compute disparity map
        disp = compute_utils.compute_disparity(left=left_data, right=right_data,
                                               min_disp=min_disp,
                                               num_disp=num_disp,
                                               block_size=block_size,
                                               P1=P1,
                                               P2=P2,
                                               disp12MaxDiff=disp12MaxDiff,
                                               preFilterCap=preFilterCap,
                                               uniquenessRatio=uniquenessRatio,
                                               speckleWindowSize=speckleWindowSize,
                                               speckleRange=speckleRange,
                                               mode=select_mode)

        # Display disparity map
        disparity_map_0_1 = cv2.normalize(disp, None, 0, 1, cv2.NORM_MINMAX)
        disp_container.image(disparity_map_0_1, caption="Disparity Map", use_column_width=True)

        # compute depth map
        depth_map = compute_utils.compute_depth(disp, baseline, fx)
        # depth_map_0_1 = cv2.normalize(depth_map, None, 0, 1, cv2.NORM_MINMAX)
        depth_container.image(depth_map, caption="Depth Map", use_column_width=True, clamp=True)
