"""
以图搜图：结构相似性（Structural Similarity，简称SSIM算法）查找相似图像的原理与实现
实验环境：Win10 | python 3.9.13 | OpenCV 4.4.0 | numpy 1.21.1 | Matplotlib 3.7.1
实验时间：2024-01-23
实例名称：SSIM_v1.0.py
"""


"""
1. 图像预处理：读取原始图像与匹配图像，并进行图像灰度处理。若两图有宽高差异，则调整图像维度
2. 计算结构相似度：计算两个灰度图像之间的结构相似度
3. 图像阈值处理：对差异图像进行阈值处理，得到一个二值化图像
4. 查找图像轮廓：在经过阈值处理后的图像中查找轮廓，并将找到的轮廓绘制在一个新的图像上
5. 绘制图像轮廓：检测到的轮廓周围放置矩形，并将处理后的图像进行展示
"""

"""
SSIM（Structural Similarity），结构相似性，是一种衡量两幅图像相似度的指标。SSIM使用的两张图像中，一张为未经压缩的无失真图像，另一张为失真后的图像。
SSIM（Structural Similarity），结构相似性，是一种衡量两幅图像相似度的指标。SSIM算法主要用于检测两张相同尺寸的图像的相似度、或者检测图像的失真程度。
原论文中，SSIM算法主要通过分别比较两个图像的亮度，对比度，结构，然后对这三个要素加权并用乘积表示。

SSIM是一种全参考的图像质量评价指标，分别从亮度、对比度、结构三个方面度量图像相似性。SSIM取值范围[0, 1]，值越大，表示图像失真越小。
在实际应用中，可以利用滑动窗将图像分块，令分块总数为N，考虑到窗口形状对分块的影响，采用高斯加权计算每一窗口的均值、方差以及协方差，然后计算对应块的结构相似度SSIM，最后将平均值作为两图像的结构相似性度量，即平均结构相似性SSIM。
"""
import cv2
import time
import numpy as np
from skimage.metrics import structural_similarity

time_start = time.time()

# 目标图像素材库文件夹路径
database_dir = '../../P0_Doc/img_data/'

# 读取查询图像和数据库中的图像
# img1_path = database_dir + 'iphone15-001.jpg'
# img2_path = database_dir + 'iphone15-002.jpg'
img1_path = database_dir + 'car-101.jpg'
img2_path = database_dir + 'car-102.jpg'

# 读取图像
img1 = cv2.imread(img1_path)
img2 = cv2.imread(img2_path)

# 将图像转换为灰度图像
img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# 检查图像形状，保证两个图像必须具有相同的尺寸，即相同的高度和宽度
if img1_gray.shape != img2_gray.shape:
    # 调整图像大小，使它们具有相同的形状
    img2_gray = cv2.resize(img2_gray, (img1_gray.shape[1], img1_gray.shape[0]))

# 计算两个图像之间的结构相似性指数（Structural Similarity Index，简称SSIM）的函数
(score, diff_img) = structural_similarity(img1_gray, img2_gray, full=True)
# (score, diff_img) = structural_similarity(img1_gray, img2_gray, win_size=101, full=True)
# 打印结构相似性指数和差异图像的信息
print(f"两个灰度图像之间的相似性指数：{score}")
print(f"两个灰度图像之间的图像结构差异：\n{diff_img}")
