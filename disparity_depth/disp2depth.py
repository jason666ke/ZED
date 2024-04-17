import cv2

import compute_utils

disp = cv2.imread('image/teddy-png-2/teddy/disp2.png')

width = disp.shape[1]
height = disp.shape[0]

fx = 260.192
baseline = 119.847

depth = compute_utils.compute_depth(disp, baseline, fx)

cv2.imshow("Depth", depth)

while True:
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cv2.destroyAllWindows()