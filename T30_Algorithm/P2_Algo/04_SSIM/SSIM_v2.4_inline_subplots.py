"""
以图搜图：结构相似性（Structural Similarity，简称SSIM算法）查找相似图像的原理与实现
实验环境：Win10 | python 3.9.13 | OpenCV 4.4.0 | numpy 1.21.1 | Matplotlib 3.7.1
实验时间：2024-01-23
实验目的：使用SSIM查找两图的结构相似性，并找出两图差异
实例名称：SSIM_v1.4_inline_subplots.py
"""

import os
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity
from matplotlib.font_manager import FontProperties

time_start = time.time()

# 目标图像素材库文件夹路径
database_dir = '../../P0_Doc/'
# 字体路径
font_path = database_dir + 'fonts/chinese_cht.ttf'

# 读取查询图像和数据库中的图像
img1_path = database_dir + 'img_data/apple-101.jpg'
img2_path = database_dir + 'img_data/apple-102.jpg'
img1_path = database_dir + 'img_data/car-101.jpg'
img2_path = database_dir + 'img_data/car-102.jpg'

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
# 打印结构相似性指数和差异图像的信息
print(f"图像2：{os.path.basename(img2_path)} 与 图像1：{img1_path} 的相似性指数：{score}")
# print(f"图像2：{os.path.basename(img2_path)} 与 图像1：{img1_path} 的图像结构差异：\n{diff_img}")

# 将差异图像的像素值缩放到 [0, 255] 范围，并转换数据类型为 uint8，以便显示
diff_img = (diff_img * 255).astype("uint8")

# 设置 Matplotlib 图像和标题，一行三列水平拼接灰度图像1、灰度图像2、灰度差异图像
fig, axs = plt.subplots(3, 3, figsize=(15, 5))
# 设置中文字体
font = FontProperties(fname=font_path, size=12)

# 在第一个子图中显示灰度图像1
axs[0][0].imshow(img1_gray, cmap='gray')
axs[0][0].set_title('灰度图像1', fontproperties=font)
# 在第二个子图中显示灰度图像2
axs[0][1].imshow(img2_gray, cmap='gray')
axs[0][1].set_title('灰度图像2', fontproperties=font)
# 在第三个子图中显示灰度差异图像
axs[0][2].imshow(diff_img, cmap='gray')
axs[0][2].set_title(f'灰度差异图像，相似性指数：{score}', fontproperties=font)


# 将差异图像进行阈值分割，返回一个经过阈值处理后的二值化图像
# 返回值有两个，第一个是阈值，第二个是二值化图像，这里只取第二个元素
img_threshold = cv2.threshold(diff_img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

# 打印差异图像进行阈值分割后的二值化图像
# print(f"img_threshold: {img_threshold}")


# 在经过阈值处理后的二值化图像中查找轮廓，并将找到的轮廓绘制在一个黑色图像上，使得图像中的轮廓变为白色
# cv2.findContours：用于查找图像中的轮廓
# 返回两个值：img_contours 包含检测到的轮廓，img_hierarchy 包含轮廓的层次结构信息
img_contours, img_hierarchy = cv2.findContours(img_threshold.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 打印检测到的轮廓信息
# print(f"img contours: {img_contours}")
# print(f"img img_hierarchy: {img_hierarchy}")


# 轮廓提取：差异图像-阈值分割-二值化图像-轮廓提取（黑底白线）
# 创建一个与阈值处理后的图像相同大小的黑色图像
img_new = np.zeros(img_threshold.shape, np.uint8)
# cv2.drawContours 在新图像上绘制轮廓，将找到的轮廓信息画用指定颜色出来，这里使用的是白色轮廓，轮廓的线宽为1
cv2.drawContours(img_new, img_contours, -1, (255, 255, 255), 1)


# 第二行用两列水平拼接二值化图像（黑底白边）、灰度差异图像
# 在第一个子图中显示二值化图像（黑底白边）
axs[1][0].imshow(img_threshold, cmap='gray')
axs[1][0].set_title('差异图像-阈值分割-二值化图像（黑底白边）', fontproperties=font)

# 在第二个子图中显示绘制图像轮廓（黑底白线）
axs[1][1].imshow(img_new, cmap='gray')
axs[1][1].set_title('差异图像-阈值分割-二值化图像-轮廓提取（黑底白线）', fontproperties=font)


# 标记差异：在检测到的轮廓差异点放置矩形进行标记，并将处理后的两图差异点进行展示
# 遍历检测到的轮廓列表，在区域周围放置矩形
for ele in img_contours:
    # 使用 cv2.boundingRect 函数计算轮廓的垂直边界最小矩形，得到矩形的左上角坐标 (x, y) 和矩形的宽度 w、高度 h
    (x, y, w, h) = cv2.boundingRect(ele)
    # 使用 cv2.rectangle 函数在原始图像 img1 上画出垂直边界最小矩形，矩形的颜色为绿色 (0, 255, 0)，线宽度为2
    cv2.rectangle(img1, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # 使用 cv2.rectangle 函数在原始图像 img2 上画出垂直边界最小矩形，矩形的颜色为绿色 (0, 255, 0)，线宽度为2
    cv2.rectangle(img2, (x, y), (x + w, y + h), (0, 255, 0), 2)


time_end = time.time()
print(f"耗时：{time_end - time_start}")

# 第三行用两列水平拼接二值化图像（黑底白边）、灰度差异图像
# 原图显示差异
axs[2][0].imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
axs[2][0].set_title('原图1', fontproperties=font)
axs[2][1].imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
axs[2][1].set_title('原图2', fontproperties=font)

# 显示 Matplotlib 图像
plt.show()