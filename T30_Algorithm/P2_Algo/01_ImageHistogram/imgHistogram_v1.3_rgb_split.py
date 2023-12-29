"""
以图搜图：图像直方图（Image Histogram）查找相似图像的原理与实现
测试环境：win10 | python 3.9.13 | OpenCV 4.4.0 | numpy 1.21.1 | matplotlib 3.7.1
实验时间：2023-10-27
实例名称：imgHistogram_v1.3_rgb_split.py
"""

# ---------------------------------------------------------------------------------------------------------------------
# 基础测试：绘制线图子图展示各通道的直方图灰度趋势
# ---------------------------------------------------------------------------------------------------------------------

import cv2
import matplotlib.pyplot as plt

# 目标图像素材库文件夹路径
database_dir = '../../P0_Doc/'
# 文件路径
img_path = database_dir + 'img_data/car-101.jpg'

# 读取图像：默认使用BGR加载图像
img = cv2.imread(img_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 分离通道：将彩色图像分离成各个通道（R、G、B），然后分别绘制它们的直方图
img_b, img_g, img_r = cv2.split(img)

# 计算各通道的直方图
hist_bgr = cv2.calcHist([img], [0], None, [256], [0, 256])
hist_rgb = cv2.calcHist([img_rgb], [0], None, [256], [0, 256])
hist_b = cv2.calcHist([img_b], [0], None, [256], [0, 256])
hist_g = cv2.calcHist([img_g], [0], None, [256], [0, 256])
hist_r = cv2.calcHist([img_r], [0], None, [256], [0, 256])

# 绘制线图子图展示各通道的直方图灰度趋势
plt.subplot(151)
plt.plot(hist_bgr, color='orange')
plt.title('BGR Histogram')

plt.subplot(152)
plt.plot(hist_rgb, color='purple')
plt.title('RGB Histogram')

plt.subplot(153)
plt.plot(hist_b, color='b')
plt.title('Blue Histogram')

plt.subplot(154)
plt.plot(hist_g, color='g')
plt.title('Green Histogram')

plt.subplot(155)
plt.plot(hist_r, color='r')
plt.title('Red Histogram')

plt.show()