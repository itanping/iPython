"""
以图搜图：差值哈希算法（Difference Hash Algorithm，简称dHash）的原理与实现
测试环境：win10 | python 3.9.13 | OpenCV 4.4.0 | numpy 1.21.1
实验时间：2023-10-25
"""

# ---------------------------------------------------------------------------------------------------------------------
# 测试：图像每一步的变换过程
# ---------------------------------------------------------------------------------------------------------------------

import cv2
import numpy as np
import matplotlib.pyplot as plt

# 测试图片路径
img_path = 'img_test/apple-01.jpg'
 
# 通过OpenCV加载图像
img = cv2.imread(img_path)
plt.imshow(img, cmap='gray')
plt.show()

# 通道重排，从BGR转换为RGB
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img_rgb, cmap='gray')
plt.show()

# 缩小图像：使用OpenCV的resize函数将图像缩放为9x8像素，采用Cubic插值方法进行图像重采样
img_resize = cv2.resize(img_rgb, (9, 8), cv2.INTER_CUBIC)

# 打印 img.shape 可以获取图像的形状信息，即 (行数, 列数, 通道数)
# 通道数：灰度图像通道数为 1，彩色图像通道数为 3
print(img_resize.shape)
plt.imshow(img_resize, cmap='gray')
plt.show()

# 图像灰度化：将彩色图像转换为灰度图像。减少计算量。
img_gray = cv2.cvtColor(img_resize, cv2.COLOR_BGR2GRAY)

# 打印出了灰度图像的行数和列数，因为灰度图像只有一个通道，所以不会显示通道数
print(img_gray.shape)
plt.imshow(img_gray, cmap='gray')
plt.show()