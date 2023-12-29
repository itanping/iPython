"""
以图搜图：图像直方图（Image Histogram）查找相似图像的原理与实现
测试环境：win10 | python 3.9.13 | OpenCV 4.4.0 | numpy 1.21.1 | matplotlib 3.7.1
实验时间：2023-10-27
实例名称：imgHistogram_v1.4_rgb_one.py
"""

# ---------------------------------------------------------------------------------------------------------------------
# 混合测试：计算各通道的直方图
# ---------------------------------------------------------------------------------------------------------------------

import cv2
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# 目标图像素材库文件夹路径
database_dir = '../../P0_Doc/'
# 文件路径
img_path = database_dir + 'img_data/car-101.jpg'
# 字体路径
font_path = database_dir + 'fonts/chinese_cht.ttf'

# 读取图像
img = cv2.imread(img_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 分离通道：将彩色图像分离成各个通道（R、G、B），然后分别绘制它们的直方图
img_b, img_g, img_r = cv2.split(img)

# 计算各通道的直方图（依次为 BGR、RGB、0：蓝色通道，1：绿色通道，2：红色通道）
hist_bgr = cv2.calcHist([img], [0], None, [256], [0, 256])
hist_rgb = cv2.calcHist([img_rgb], [0], None, [256], [0, 256])

hist_b = cv2.calcHist([img_b], [0], None, [256], [0, 256])
hist_g = cv2.calcHist([img_g], [0], None, [256], [0, 256])
hist_r = cv2.calcHist([img_r], [0], None, [256], [0, 256])

hist_0 = cv2.calcHist([img], [0], None, [256], [0, 256])
hist_1 = cv2.calcHist([img], [1], None, [256], [0, 256])
hist_2 = cv2.calcHist([img], [2], None, [256], [0, 256])

# 绘制多线线图，展示各通道的直方图灰度趋势
plt.plot(hist_bgr, color='orange', label='BGR')
plt.plot(hist_rgb, color='purple', label='RGB')

plt.plot(hist_b, color='blue', label='Channel Blue')
plt.plot(hist_0, color='blue', label='Channel 0')

plt.plot(hist_g, color='green', label='Channel Green')
plt.plot(hist_1, color='green', label='Channel 1')

plt.plot(hist_r, color='red', label='Channel Red')
plt.plot(hist_2, color='red', label='Channel 2')

# 设置中文字体
font = FontProperties(fname=font_path, size=14)
plt.title('Histogram（直方图）', fontproperties=font)
plt.xlabel('Pixel Value（像素值）', fontproperties=font)
plt.ylabel('Frequency（频率）', fontproperties=font)

# 添加图例
plt.legend()
# 显示图像
plt.show()