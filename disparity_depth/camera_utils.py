import open3d as o3d
import pyzed.sl as sl


def connect(init):
    zed = sl.Camera()
    status = zed.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        print(repr(status))
        return None
    else:
        return zed


def get_camera_intrinsics(camera):
    """
    Get the camera intrinsic matrix and focal length and baseline
    :param camera: zed camera object
    :return: camera_intrinsics, focal_left_x, baseline_mm
    """
    camera_info = camera.get_camera_information()
    calibration_params = camera_info.camera_configuration.calibration_parameters

    # 焦距和基线距离
    focal_left_x = calibration_params.left_cam.fx  # 焦距（像素单位）
    baseline_mm = calibration_params.get_camera_baseline()  # 基线距离(毫米为单位)
    print("Left cam fx: {0} pixel".format(focal_left_x))
    print("Baseline: {0} millimeters".format(baseline_mm))

    # 摄像头内参
    width = calibration_params.left_cam.image_size.width
    height = calibration_params.left_cam.image_size.height
    fx = calibration_params.left_cam.fx
    fy = calibration_params.left_cam.fy
    cx = calibration_params.left_cam.cx
    cy = calibration_params.left_cam.cy
    camera_intrinsics = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

    print("fx: {0} pixel".format(focal_left_x))
    print("fy: {0} pixel".format(fy))
    print("cx: {0} pixel".format(cx))
    print("cy: {0} pixel".format(cy))
    print("Image size: ({0} {1})".format(height, width))

    return camera_intrinsics, focal_left_x, baseline_mm


def get_image_size(camera):
    camera_info = camera.get_camera_information()
    calibration_params = camera_info.camera_configuration.calibration_parameters
    size = calibration_params.left_cam.image_size

    return size
