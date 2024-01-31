import numpy as np
import pyzed.sl as sl
import cv2

# left = cv2.imread('image/left_eye.png')
left = cv2.imread('image/teddy-png-2/teddy/im2.png')
# right = cv2.imread('image/right_eye.png')
right = cv2.imread('image/teddy-png-2/teddy/im6.png')

if left is None or right is None:
    print("Could not read the image files")
    exit()

print(left.shape, right.shape)

print("Left and Right have same size: ", left.shape == right.shape)

height, width = left.shape[:2]
print("Height and Width: ({0}, {1})".format(height, width))

# 将图像转换为灰度图像
left_gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
right_gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

# 立体匹配对象
stereo = cv2.StereoSGBM.create(numDisparities=16, blockSize=15)

disparity = stereo.compute(left_gray, right_gray)

# 归一化视差图以便于显示
# disparity_normalized = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
disparity_normalized = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

cv2.imwrite('image/disparity.png', disparity_normalized)

cv2.imshow('left', left_gray)
cv2.imshow('disparity', disparity_normalized)

cv2.waitKey()
cv2.destroyAllWindows()
