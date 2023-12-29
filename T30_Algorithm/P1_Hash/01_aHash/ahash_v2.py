"""
以图搜图：均值哈希算法（Average Hash Algorithm）的原理与实现
测试环境：win10 | python 3.9.13 | OpenCV 4.4.0 | numpy 1.21.1
实验时间：2023-10-20
"""

# ---------------------------------------------------------------------------------------------------------------------
# 测试图像灰度
# 为什么使用了 cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) ，但显示出来图像是彩色的？
# ---------------------------------------------------------------------------------------------------------------------

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# # 测试图片路径
# img_path = 'img_test/apple-01.jpg'

# 获取当前脚本文件的路径
script_dir = os.path.dirname(__file__)
print(script_dir)

# 测试图片全路径
img_path = os.path.join(script_dir, 'img_test/apple-01.jpg')
print(img_path)

# 通过OpenCV加载图像
img = cv2.imread(img_path)

# 通道重排，从BGR转换为RGB
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 使用OpenCV的resize函数将图像缩放为8x8像素，采用Cubic插值方法
# cv2.INTER_NEAREST：最近邻插值，也称为最近邻算法。它简单地使用最接近目标像素的原始像素的值。虽然计算速度快，但可能导致图像质量下降。
# cv2.INTER_LINEAR：双线性插值，通过对最近的4个像素进行线性加权来估计目标像素的值。比最近邻插值更精确，但计算成本略高。
# cv2.INTER_CUBIC：双三次插值，使用16个最近像素的加权平均值来估计目标像素的值。通常情况下，这是一个不错的插值方法，适用于图像缩小。
# cv2.INTER_LANCZOS4：Lanczos插值，一种高质量的插值方法，使用Lanczos窗口函数。通常用于缩小图像，以保留图像中的细节和纹理。
img_resize = cv2.resize(img_rgb, (8, 8), cv2.INTER_CUBIC)

# 图像灰度化：将彩色图像转换为灰度图像。
img_gray = cv2.cvtColor(img_resize, cv2.COLOR_BGR2GRAY)

# 灰度形式查看图像
plt.imshow(img_gray, cmap='gray')
# 显示图像
plt.show()