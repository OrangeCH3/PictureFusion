#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    : run2.py
import numpy as np
import cv2
import math


def img1_extract_point(img, threshold):
    img = img.copy()
    img[img < threshold] = 0
    img[img > threshold] = 255
    _, labels = cv2.connectedComponents(img)
    _, labels, stats, centroids = cv2.connectedComponentsWithStats(img)
    centroids = np.array(centroids, dtype=int).reshape(-1, 2)
    # for j in centroids:
    #     y, x = j
    #     img[x, y] = 255
    return centroids[[1, 2], :]


def img2_extract_point(img, threshold):  # 提取图二特征点
    img = img.copy()
    img[img < threshold] = 0
    img[img > threshold] = 255
    _, labels = cv2.connectedComponents(img)
    _, labels, stats, centroids = cv2.connectedComponentsWithStats(img)
    centroids = np.array(centroids, dtype=int).reshape(-1, 2)
    for j in centroids:
        y, x = j
        img[x, y] = 255
    return centroids[[0, 2], :]


def whirl_1(img, angle, scaling_ratio, shape):  # 旋转图一
    m, n = shape
    img1 = cv2.copyMakeBorder(img, 200, 200, 200, 0, cv2.BORDER_CONSTANT, value=[0])
    print(img1.shape, scaling_ratio, )
    # m, n = int(m / scaling_ratio), int(n / scaling_ratio)
    # M = cv2.getRotationMatrix2D((img2_centroids[0, 1], img2_centroids[0, 0]), angle, 1)
    M = cv2.getRotationMatrix2D((img1_centroids[1, 1], img1_centroids[1, 0]), angle, scaling_ratio)
    dst = cv2.warpAffine(img1, M, (m, n * 2))

    # 显示图形
    cv2.imwrite('save_img1.jpg', dst)
    cv2.namedWindow("Open-CV2", 0)
    cv2.imshow("Open-CV2", dst)  # 显示图片窗口
    cv2.waitKey(0)  # 窗口显示时间
    cv2.destroyAllWindows()


def whirl_2(img, angle, scaling_ratio, shape):  # 旋转图一
    m, n = shape
    img1 = cv2.copyMakeBorder(img, 50, 200, 200, 0, cv2.BORDER_CONSTANT, value=[0])
    print(img1.shape, scaling_ratio, )
    # m, n = int(m / scaling_ratio), int(n / scaling_ratio)
    # M = cv2.getRotationMatrix2D((img2_centroids[0, 1], img2_centroids[0, 0]), angle, 1)
    M = cv2.getRotationMatrix2D((img1_centroids[1, 1], img1_centroids[1, 0]), angle, scaling_ratio)
    dst = cv2.warpAffine(img1, M, (m * 3, n * 3))

    # 显示图形
    cv2.imwrite('save——img2.jpg', dst)
    cv2.namedWindow("Open-CV2", 0)
    cv2.imshow("Open-CV2", dst)  # 显示图片窗口
    cv2.waitKey(0)  # 窗口显示时间
    cv2.destroyAllWindows()


if __name__ == "__main__":
    img1 = cv2.imread('1.01.jpg', 0)
    img2 = cv2.imread('1.02.jpg', 0)

    img1_centroids = img1_extract_point(img1, 240)  # 图像一的特征点
    img2_centroids = img2_extract_point(img2, 60)  # 图像二的特征点
    print(img1_centroids, img2_centroids)

    distance1 = np.linalg.norm(img1_centroids[0] - img1_centroids[1])  # 图像一特征点之间的欧氏距离
    distance2 = np.linalg.norm(img2_centroids[0] - img2_centroids[1])  # 图像二特征点之间的欧氏距离
    scaling_ratio = distance1 / distance2  # 缩放比例

    # img1 倾斜角度
    dx = img1_centroids[0, 0] - img1_centroids[1, 0]  # 83-34=49
    dy = img1_centroids[0, 1] - img1_centroids[1, 1]  # 109-128 =-19
    angle_1 = math.atan2(dy, dx) * 180.0 / math.pi
    # img2 倾斜角度
    dx = img2_centroids[0, 0] - img2_centroids[1, 0]
    dy = img2_centroids[0, 1] - img2_centroids[1, 1]
    angle_2 = math.atan2(dy, dx) * 180.0 / math.pi

    # 旋转
    angle = (-angle_2 + angle_1)

    dst1 = whirl_1(img1, angle=angle, scaling_ratio=0.3, shape=img1.shape)  # 旋转图1
    dst2 = whirl_2(img2, angle=0, scaling_ratio=scaling_ratio * 0.3, shape=img2.shape)  # 旋转图2
    
