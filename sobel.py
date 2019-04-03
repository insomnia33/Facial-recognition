import argparse
import cv2
import numpy as np

image = cv2.imread("./arquivo/ex1.jpg")
sobel = cv2.Sobel(image, cv2.CV_16S,1,0, 5)
sobel = cv2.convertScaleAbs(sobel)

output = np.hstack((image, sobel))

cv2.imshow("Original", output)

cv2.waitKey()