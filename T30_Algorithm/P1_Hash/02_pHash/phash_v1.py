"""
以图搜图：感知哈希算法（Perceptual hash algorithm）的原理与实现
测试环境：win10 | python 3.9.13 | OpenCV 4.4.0 | numpy 1.21.1
实验时间：2023-10-22
"""

# 以图搜图：感知哈希算法（Perceptual hash algorithm）的原理与实现。
# 
# 感知哈希算法借助离散余弦变换（Discrete Cosine Transform，DCT）来提取图像的频率特征。
# 它首先将图像转换为灰度图像，并调整图像的大小为固定的尺寸（如32x32像素）。
# 然后，对调整后的图像应用DCT，并保留低频分量。
# 接下来，根据DCT系数的相对大小，将图像转换为一个二进制哈希值。
# 通过计算两个图像哈希值的汉明距离，可以衡量图像的相似度。

# ---------------------------------------------------------------------------------------------------------------------
# 测试原图读取
# 为什么通过 cv2.imread(img_path) 加载的图像，显示出来之后，原图由红色变成了蓝色？
#
# 测试图像灰度
# 为什么使用了 cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) ，但显示出来图像是彩色的？
# ---------------------------------------------------------------------------------------------------------------------

import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

print("Python 版本:", sys.version)
print("OpenCV 版本:", cv2.__version__)
print("numpy 版本:", np.__version__)

# 测试图片路径
img_path = 'img_test/apple-01.jpg'
 
# 通过OpenCV加载图像
img = cv2.imread(img_path)

# 通道重排，从BGR转换为RGB
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 缩小图像：使用OpenCV的resize函数将图像缩放为32x32像素，采用Cubic插值方法进行图像重采样
img_32 = cv2.resize(img, (32, 32), cv2.INTER_CUBIC)

# 图像灰度化：将彩色图像转换为灰度图像。减少计算量。
img_gray = cv2.cvtColor(img_32, cv2.COLOR_BGR2GRAY)


plt.imshow(img_32, cmap='gray')
plt.show()