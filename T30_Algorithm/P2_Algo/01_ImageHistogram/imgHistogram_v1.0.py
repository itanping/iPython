"""
以图搜图：图像直方图（Image Histogram）查找相似图像的原理与实现
测试环境：win10 | python 3.9.13 | OpenCV 4.4.0 | numpy 1.21.1 | matplotlib 3.7.1
实验时间：2023-10-27
实例名称：imgHistogram_v1.0.py
"""

# ---------------------------------------------------------------------------------------------------------------------
# 基础测试：使用直方图绘制一张图像
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
# 计算直方图
img_hist = cv2.calcHist([img], [0], None, [256], [0, 256])
print(img_hist)

# 设置中文字体
font = FontProperties(fname=font_path, size=14)

# 绘制线图
# plt.plot(img_hist)

# 绘制直方图
# bins 参数指定了直方图中的柱子数量
plt.hist(img_hist, bins=len(img_hist), color='black', alpha=0.8)

plt.title('Histogram（直方图）', fontproperties=font)
plt.xlabel('Pixel Value（像素值）', fontproperties=font)
plt.ylabel('Frequency（频率）', fontproperties=font)

# 像素分布可视化
plt.show()