"""
以图搜图：图像直方图（Image Histogram）查找相似图像的原理与实现
测试环境：win10 | python 3.9.13 | OpenCV 4.4.0 | numpy 1.21.1 | matplotlib 3.7.1
实验时间：2023-10-27
实例名称：imgHistogram_v1.1_rgb.py
"""

# ---------------------------------------------------------------------------------------------------------------------
# 基础测试
# ---------------------------------------------------------------------------------------------------------------------

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

"""
cv2.calcHist 函数用于计算图像的直方图。以下是它的主要参数和说明：
    images：要计算直方图的图像。它以方括号的形式传递，允许计算多个图像的直方图。例如，[img] 表示计算单个图像的直方图，[img1, img2] 表示计算两个图像的直方图。
    channels：指定要考虑的通道。这是一个通道索引列表，用于选择要计算直方图的通道。在OpenCV中，通常情况下，通道0对应于蓝色（B），通道1对应于绿色（G），通道2对应于红色（R）。如果要考虑所有通道，可以使用 [0, 1, 2]，而对于单通道，只会考虑图像的灰度信息，而不考虑颜色信息。使用 [0] 即可表示灰色通道，与 [1] 或 [2] 效果等同。
    mask：可选参数，用于指定一个掩码图像，以便只计算掩码中非零元素对应的像素值。如果不需要掩码，可以将其设置为 None。
    histSize：指定直方图的 bin 数量，即要计算的直方图的维度。它通常以方括号形式传递，例如 [256] 表示每个通道有 256 个 bin。
    ranges：指定像素值的范围。通常以方括号形式传递，例如 [0, 256] 表示像素值的范围从 0 到 255。对于彩色图像，通常设置为 [0, 256, 0, 256, 0, 256]，表示三个通道的范围。
"""

# 目标图像素材库文件夹路径
database_dir = '../../P0_Doc/'
# 文件路径
img_path = database_dir + 'img_data/car-101.jpg'
# 字体路径
font_path = database_dir + 'fonts/chinese_cht.ttf'
img = cv2.imread(img_path)

# 计算三个通道的直方图
img_hist0 = cv2.calcHist([img], [0], None, [256], [0, 256])
# # 设置中文字体
# font = FontProperties(fname="./fonts/chinese_cht.ttf", size=14)
# # 绘制直方图
# plt.plot(img_hist0)
# plt.title('Histogram（直方图）', fontproperties=font)
# plt.xlabel('Pixel Value（像素值）', fontproperties=font)
# plt.ylabel('Frequency（频率）', fontproperties=font)
# plt.show()

img_hist1 = cv2.calcHist([img], [1], None, [256], [0, 256])
# # 设置中文字体
# font = FontProperties(fname="./fonts/chinese_cht.ttf", size=14)
# # 绘制直方图
# plt.plot(img_hist1)
# plt.title('Histogram（直方图）', fontproperties=font)
# plt.xlabel('Pixel Value（像素值）', fontproperties=font)
# plt.ylabel('Frequency（频率）', fontproperties=font)
# plt.show()

img_hist2 = cv2.calcHist([img], [2], None, [256], [0, 256])
# # 设置中文字体
# font = FontProperties(fname="./fonts/chinese_cht.ttf", size=14)
# # 绘制直方图
# plt.plot(img_hist2)
# plt.title('Histogram（直方图）', fontproperties=font)
# plt.xlabel('Pixel Value（像素值）', fontproperties=font)
# plt.ylabel('Frequency（频率）', fontproperties=font)
# plt.show()

# 绘制直方图
plt.plot(img_hist0, color='blue', label='Channel 0 (Blue)')
plt.plot(img_hist1, color='green', label='Channel 1 (Green)')
plt.plot(img_hist2, color='red', label='Channel 2 (Red)')

# 设置中文字体
font = FontProperties(fname=font_path, size=14)
# 绘制线图
plt.title('Histogram（直方图）', fontproperties=font)
plt.xlabel('Pixel Value（像素值）', fontproperties=font)
plt.ylabel('Frequency（频率）', fontproperties=font)

# 添加图例
plt.legend()

# 显示图像
plt.show()
