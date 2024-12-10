# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 17:33:06 2020

@author: lewis
"""

import cv2 #Open CV
import numpy as np

img = cv2.imread('Example_Ortho_5k.jpg')  # Loading an image

#HSV is the Hue Saturation Value color space that can be useful in some image processing applications but this script doesn't end up using it
hsv_frame = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) # Changing the Color space of the image


#Display each of the color channels (this can be done for RGB images too)
channels = cv2.split(hsv_frame)
cv2.imshow("Hue Channel", channels[0])  
cv2.imshow("Saturation Channel", channels[1])
cv2.imshow("Value Channel", channels[2])

# Using thresholds to only keep a certain color space in the image. In this kase 
low_cut = np.array([168,174,169])
high_cut = np.array([86, 107, 98])
mask = cv2.inRange(img, high_cut, low_cut)

cv2.imshow("Original", img)
cv2.imshow("Masked", mask)

#########################
# Method from stack overflow to remove the noise seen in "Masked"
# https://stackoverflow.com/questions/48681465/how-do-i-remove-the-dots-noise-without-damaging-the-text/48682812
_, blackAndWhite = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(blackAndWhite, None, None, None, 8, cv2.CV_32S)
sizes = stats[1:, -1]
img2 = np.zeros((labels.shape), np.uint8)

for i in range(0, nlabels - 1):
    if sizes[i] >= 100:
        img2[labels == i+1] = 255
        
res = cv2.bitwise_not(img2)
##########################


kernel = np.ones((5,5), np.uint8)

# This is called a Morphological Transformation. In this case it is "opening" which is erosion followed by dilation
#  https://docs.opencv.org/3.4.0/d9/d61/tutorial_py_morphological_ops.html
opening = cv2.morphologyEx(res, cv2.MORPH_OPEN, kernel)

cv2.imshow("Final", res)
cv2.imshow("Final_Morph", opening)

# Very simple edge detection
edges = cv2.Canny(opening, 20, 30)
cv2.imshow("Edges", edges)

# This is how to quit stuff without it acting weird
cv2.waitKey(0)
cv2.destroyAllWindows()