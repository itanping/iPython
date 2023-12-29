"""
以图搜图：图像直方图（Image Histogram）查找相似图像的原理与实现
测试环境：win10 | python 3.9.13 | OpenCV 4.4.0 | matplotlib 3.7.1
实验时间：2023-10-27
实例名称：imgHistogram_v2.5_green.py
"""

# ---------------------------------------------------------------------------------------------------------------------
# 两图测试
# ---------------------------------------------------------------------------------------------------------------------

import cv2
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

def calcHist(img1_path, img2_path):
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    img1_hist = cv2.calcHist([img1], [1], None, [256], [0, 256])
    img2_hist = cv2.calcHist([img2], [1], None, [256], [0, 256])
    similarity = cv2.compareHist(img1_hist, img2_hist, cv2.HISTCMP_CORREL)
    print(f"图像 {img2_path} 与目标图像 {img1_path} 的相似度（HISTCMP_CORREL/相关性）：", similarity)

    # 设置中文字体
    font = FontProperties(fname=font_path, size=14)
    # 绘制直方图
    plt.plot(img1_hist)
    plt.title('Histogram（直方图）', fontproperties=font)
    plt.xlabel('Pixel Value（像素值）', fontproperties=font)
    plt.ylabel('Frequency（频率）', fontproperties=font)
    plt.show()

    return similarity

# 目标图像素材库文件夹路径
database_dir = '../../P0_Doc/'
# 字体路径
font_path = database_dir + 'fonts/chinese_cht.ttf'
# 文件路径
img1_path = database_dir + 'img_data/car-101.jpg'
img2_path = database_dir + 'img_data/car-102.jpg'

calcHist(img1_path, img2_path)