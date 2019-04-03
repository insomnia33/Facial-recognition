import argparse
import cv2
import numpy as np

image = cv2.imread("./arquivo/logo.jpg")
laplacian = cv2.Laplacian(image, cv2.CV_16S, 3)
laplacian = cv2.convertScaleAbs(laplacian)

output = np.hstack((image, laplacian))

cv2.imshow("Original", output)

cv2.waitKey()