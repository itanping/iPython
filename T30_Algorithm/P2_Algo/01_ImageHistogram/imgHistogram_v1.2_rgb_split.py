"""
以图搜图：图像直方图（Image Histogram）查找相似图像的原理与实现
测试环境：win10 | python 3.9.13 | OpenCV 4.4.0 | numpy 1.21.1 | matplotlib 3.7.1
实验时间：2023-10-27
实例名称：imgHistogram_v1.2_rgb_split.py
"""

# ---------------------------------------------------------------------------------------------------------------------
# 基础测试：将彩色图像分离成各个通道（R、G、B），然后分别绘制它们的直方图
# ---------------------------------------------------------------------------------------------------------------------

import cv2
import matplotlib.pyplot as plt

# 目标图像素材库文件夹路径
database_dir = '../../P0_Doc/'
# 文件路径
img_path = database_dir + 'img_data/car-101.jpg'

# 读取图像：默认使用BGR加载图像
img = cv2.imread(img_path)
# 分离通道：将彩色图像分离成各个通道（R、G、B），然后分别绘制它们的直方图
img_b, img_g, img_r = cv2.split(img)

# 绘制子图
plt.figure(figsize=(15, 5))
# 151：表示子图位于一个 1x5 的网格中的第一个位置。如比第2张图的位置152，即一行五列第2张图
# 显示各通道的图像
plt.subplot(151)
plt.imshow(img)
plt.title('BGR (Default)')

plt.subplot(152)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('BGR TO RGB')

plt.subplot(153)
plt.imshow(cv2.cvtColor(img_b, cv2.COLOR_BGR2RGB))
plt.title('Blue Channel')

plt.subplot(154)
plt.imshow(cv2.cvtColor(img_g, cv2.COLOR_BGR2RGB))
plt.title('Green Channel')

plt.subplot(155)
plt.imshow(cv2.cvtColor(img_r, cv2.COLOR_BGR2RGB))
plt.title('Red Channel')

plt.show()