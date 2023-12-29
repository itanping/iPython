"""
以图搜图：图像直方图（Image Histogram）查找相似图像的原理与实现
测试环境：win10 | python 3.9.13 | OpenCV 4.4.0 | numpy 1.21.1
实验时间：2023-10-27
"""

# ---------------------------------------------------------------------------------------------------------------------
# 测试原图读取
# 为什么通过 cv2.imread(img_path) 加载的图像，显示出来之后，原图由红色变成了蓝色？
#
# 测试图像灰度
# 为什么使用了 cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) ，但显示出来图像是彩色的？
# ---------------------------------------------------------------------------------------------------------------------

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

def calcHist(img1, img2):
    img1 = cv2.imread(img1)
    img2 = cv2.imread(img2)

    # 将图像转换为灰度图像
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    """
    cv2.calcHist 函数用于计算图像的直方图。以下是它的主要参数和说明：
        images：要计算直方图的图像。它以方括号的形式传递，允许计算多个图像的直方图。例如，[img] 表示计算单个图像的直方图，[img1, img2] 表示计算两个图像的直方图。
        channels：指定要考虑的通道。这是一个通道索引列表，用于选择要计算直方图的通道。在OpenCV中，通常情况下，通道0对应于蓝色（B），通道1对应于绿色（G），通道2对应于红色（R）。如果要考虑所有通道，可以使用 [0, 1, 2]，而对于单通道，只会考虑图像的灰度信息，而不考虑颜色信息。使用 [0] 即可表示灰色通道，与 [1] 或 [2] 效果等同。
        mask：可选参数，用于指定一个掩码图像，以便只计算掩码中非零元素对应的像素值。如果不需要掩码，可以将其设置为 None。
        histSize：指定直方图的 bin 数量，即要计算的直方图的维度。它通常以方括号形式传递，例如 [256] 表示每个通道有 256 个 bin。
        ranges：指定像素值的范围。通常以方括号形式传递，例如 [0, 256] 表示像素值的范围从 0 到 255。对于彩色图像，通常设置为 [0, 256, 0, 256, 0, 256]，表示三个通道的范围。
    """
    img1_hist = cv2.calcHist([img1_gray], [0], None, [256], [0, 256])
    img2_hist = cv2.calcHist([img2_gray], [0], None, [256], [0, 256])
    # 设置中文字体
    font = FontProperties(fname=font_path, size=14)
    # 绘制直方图
    plt.plot(img1_hist)
    plt.title('Histogram（直方图）', fontproperties=font)
    plt.xlabel('Pixel Value（像素值）', fontproperties=font)
    plt.ylabel('Frequency（频率）', fontproperties=font)
    plt.show()

    # 归一化直方图
    hist1 = cv2.normalize(img1_hist, img1_hist, 0, 1, cv2.NORM_MINMAX)
    hist2 = cv2.normalize(img2_hist, img2_hist, 0, 1, cv2.NORM_MINMAX)
   
    # 设置中文字体
    font = FontProperties(fname=font_path, size=14)
    # 绘制直方图
    plt.plot(hist1)
    plt.title('Histogram（直方图）', fontproperties=font)
    plt.xlabel('Pixel Value（像素值）', fontproperties=font)
    plt.ylabel('Frequency（频率）', fontproperties=font)
    plt.show()

    # 计算直方图相似度
    # degree = 0
    # for i in range(len(img1_hist)):
    #     if img1_hist[i] != img2_hist[i]:
    #         degree = degree + (1 - abs(img2_hist[i] - img2_hist[i]) / max(img2_hist[i], img2_hist[i]))
    #     else:
    #         degree = degree + 1
    # degree = degree / len(img2_hist)
    # print(degree)

    # 计算直方图相似度
    # cv2.HISTCMP_CORREL 是比较直方图的方法之一，它计算两个直方图之间的相关性。相关性的值越接近1，表示两幅图像的颜色分布越相似，值越接近-1表示颜色分布越不相似，值接近0表示中等相似度。
    # distance = cv2.compareHist(img1_hist, img2_hist, cv2.HISTCMP_CORREL)

    # 计算直方图相似度：使用巴氏距离
    distance = cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)
    return distance

# 目标图像素材库文件夹路径
database_dir = '../../P0_Doc/'
# 文件路径
img1_path = database_dir + 'img_data/apple-101.jpg'
img2_path = database_dir + 'img_data/apple-102.jpg'
# 字体路径
font_path = database_dir + 'fonts/chinese_cht.ttf'

degree = calcHist(img1_path, img2_path)
print(degree)