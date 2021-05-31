# 显示图片
import numpy as np
import cv2
from matplotlib import pyplot as plt

# 读取图片内容
img1 = cv2.imread('1.01.jpg', 0)
img2 = cv2.imread('1.02.jpg', 0)

# 使用ORB特征检测器和描述符，计算关键点和描述符
orb = cv2.ORB_create(3)
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# img3 = cv2.drawKeypoints(img1, kp1, outImage=img1, color=(0, 255, 0), flags=0)
# plt.imshow(img3), plt.show()
# input()

# 暴力匹配BFMatcher，遍历描述符，确定描述符是否匹配，然后计算匹配距离并排序
# BFMatcher函数参数：
# normType：NORM_L1, NORM_L2, NORM_HAMMING, NORM_HAMMING2。
# NORM_L1和NORM_L2是SIFT和SURF描述符的优先选择，NORM_HAMMING和NORM_HAMMING2是用于ORB算法
bf = cv2.BFMatcher(normType=cv2.NORM_HAMMING2, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)
# matches是DMatch对象，具有以下属性：
# DMatch.distance - 描述符之间的距离。 越低越好。
# DMatch.trainIdx - 训练描述符中描述符的索引
# DMatch.queryIdx - 查询描述符中描述符的索引
# DMatch.imgIdx - 训练图像的索引。

# 使用plt将两个图像的匹配结果显示出来
img3 = cv2.drawMatches(img1=img1, keypoints1=kp1, img2=img2, keypoints2=kp2, matches1to2=matches, outImg=img2, flags=2)
plt.imshow(img3), plt.show()

