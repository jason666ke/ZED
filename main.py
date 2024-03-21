import pyzed.sl as sl
import torch

def main():
    # Create a Camera object
    zed = sl.Camera()

    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters()
    init_params.sdk_verbose = True

    # Open the camera
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        exit(1)

    # Get camera information (ZED serial number)
    zed_serial = zed.get_camera_information().serial_number
    zed_camara_model = zed.get_camera_information().camera_model
    print("Hello! This is my serial number: {0}".format(zed_serial))
    print("Hello! This is my camera model: {0}".format(zed_camara_model))

    # Close the camera
    zed.close()


if __name__ == "__main__":
    print(torch.__version__)
    # main()
