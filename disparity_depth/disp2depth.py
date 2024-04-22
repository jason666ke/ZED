import cv2

import compute_utils

disp = cv2.imread('image/teddy-png-2/teddy/disp2.png')

width = disp.shape[1]
height = disp.shape[0]

fx = 100
baseline = 100

depth = compute_utils.compute_depth(disp, baseline, fx)

cv2.imshow("Disp", disp)
cv2.imshow("Depth", depth)

while True:
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cv2.destroyAllWindows()