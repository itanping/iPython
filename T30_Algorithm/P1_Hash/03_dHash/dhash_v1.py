"""
以图搜图：差值哈希算法（Difference Hash Algorithm，简称dHash）的原理与实现
测试环境：win10 | python 3.9.13 | OpenCV 4.4.0 | numpy 1.21.1
实验时间：2023-10-25
"""

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

# 缩小图像：使用OpenCV的resize函数将图像缩放为9x8像素，采用Cubic插值方法进行图像重采样
img = cv2.resize(img, (9, 8), cv2.INTER_CUBIC)
print(img.shape)

img = cv2.resize(img, (9, 8), interpolation=cv2.INTER_CUBIC)
print(img.shape)

# 图像灰度化：将彩色图像转换为灰度图像。减少计算量。
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# plt.imshow(img, cmap='gray')
# plt.show()