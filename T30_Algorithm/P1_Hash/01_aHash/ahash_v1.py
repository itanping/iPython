"""
以图搜图：均值哈希算法（Average Hash Algorithm）的原理与实现
测试环境：win10 | python 3.9.13 | OpenCV 4.4.0 | numpy 1.21.1
实验时间：2023-10-20
"""

# ---------------------------------------------------------------------------------------------------------------------
# 测试原图读取
# 为什么通过 cv2.imread(img_path) 加载的图像，显示出来之后，原图由红色变成了蓝色？
# ---------------------------------------------------------------------------------------------------------------------
import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

print("Python 版本:", sys.version)
print("OpenCV 版本:", cv2.__version__)
print("numpy 版本:", np.__version__)

# 获取当前脚本文件的路径
# os.path.dirname(__file__) 用于获取当前脚本文件的目录，然后使用 os.path.join() 构建图像文件的绝对路径
script_dir = os.path.dirname(__file__)
print(script_dir)

# 测试图片全路径
img_path = os.path.join(script_dir, 'img_test/apple-01.jpg')
print(img_path)

# 测试图片路径
# img_path = 'img_test/apple-01.jpg'

# 通过OpenCV加载图像
img = cv2.imread(img_path)
plt.imshow(img)
plt.show()

# 通道重排，从BGR转换为RGB
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.show()

# 使用OpenCV的resize函数将图像缩放为8x8像素，采用Cubic插值方法
# cv2.INTER_NEAREST：最近邻插值，也称为最近邻算法。它简单地使用最接近目标像素的原始像素的值。虽然计算速度快，但可能导致图像质量下降。
# cv2.INTER_LINEAR：双线性插值，通过对最近的4个像素进行线性加权来估计目标像素的值。比最近邻插值更精确，但计算成本略高。
# cv2.INTER_CUBIC：双三次插值，使用16个最近像素的加权平均值来估计目标像素的值。通常情况下，这是一个不错的插值方法，适用于图像缩小。
# cv2.INTER_LANCZOS4：Lanczos插值，一种高质量的插值方法，使用Lanczos窗口函数。通常用于缩小图像，以保留图像中的细节和纹理。
img_resize = cv2.resize(img, (8, 8), cv2.INTER_CUBIC)

# plt.imshow(img)
plt.imshow(img_resize)
plt.show()