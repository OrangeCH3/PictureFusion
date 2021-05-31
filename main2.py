#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    : run1.py

import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('1.01.jpg', 0)

# Initiate STAR detector
orb = cv2.ORB_create(3)

# find the keypoints with ORB
kp = orb.detect(img, None)

# compute the descriptors with ORB
# kp, des = orb.compute(img, kp)
# print(kp)


# draw only keypoints location,not size and orientation
img2 = cv2.drawKeypoints(img, kp, outImage=img, color=(0, 255, 0), flags=0)
plt.imshow(img2), plt.show()
