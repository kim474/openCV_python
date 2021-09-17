import cv2
import numpy as np
import matplotlib.pyplot as plt

# blur까지 이미지 히스토그램
cam = cv2.VideoCapture(2)
cam.set(3, 480)
cam.set(4, 480)

while True:
    ret, frame = cam.read()

    if(ret):
        crop = frame[45:435, 0:480].copy()
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        cv2.imwrite("../w1_l1.jpg", blur)
        hist = cv2.calcHist([blur], [0], None, [256], [0, 256])
        cv2.imshow("plastic1_rl6_bl", blur)
        cv2.waitKey(1)
        plt.plot(hist)
        plt.show()
        break

cam.release()
cv2.destroyAllWindows()
