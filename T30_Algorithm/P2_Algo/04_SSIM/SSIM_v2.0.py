# SSIM（结构相似度度量）计算图片的相似度
"""
以图搜图：余弦相似度（Cosine Similarity）查找相似图像的原理与实现
实验环境：Win10 | python 3.9.13 | OpenCV 4.4.0 | numpy 1.21.1 | Matplotlib 3.7.1
实验时间：2023-11-30
实例名称：imgCosineSimilarity_v1.0_show.py
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
# 打印结构相似性指数和差异图像的信息
print("两个灰度图像之间的结构相似度：{}".format((score, diff_img)))

# 将差异图像的像素值缩放到 [0, 255] 范围，并转换数据类型为 uint8，以便显示
diff_img = (diff_img * 255).astype("uint8")
# 创建一个名为 "diff_img" 的窗口，并使用 cv2.WINDOW_NORMAL 设置窗口属性为可调整大小
cv2.namedWindow("diff_img", cv2.WINDOW_NORMAL)
# 在 "diff_img" 窗口中显示差异图像
cv2.imshow("diff_img", diff_img)
# 打印结构相似性指数
print("SSIM：{}".format(score))


# 找到不同的轮廓以致于可以在表示为'不同'的区域放置矩形
# 全局自适应阈值分割（二值化），返回值有两个，第一个是阈值，第二个是二值图像


# 将差异图像进行阈值分割，返回一个经过阈值处理后的二值化图像
# 返回值有两个，第一个是阈值，第二个是二值化图像，这里只取第二个元素
img_threshold = cv2.threshold(diff_img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

# 创建一个名为 "threshold" 的窗口，并使用 cv2.WINDOW_NORMAL 设置窗口属性为可调整大小
cv2.namedWindow("threshold", cv2.WINDOW_NORMAL)
# 在 "threshold" 窗口中显示经过阈值处理后的图像 img_threshold
cv2.imshow('threshold', img_threshold)


# findContours找轮廓，返回值有两个，第一个是轮廓信息，第二个是轮廓的层次信息（“树”状拓扑结构）
# cv2.RETR_EXTERNAL：只检测最外层轮廓
# cv2.CHAIN_APPROX_SIMPLE：压缩水平方向、垂直方向和对角线方向的元素，保留该方向的终点坐标，如矩形的轮廓可用4个角点表示


# 在经过阈值处理后的图像中查找轮廓，并将找到的轮廓绘制在一个黑色图像上，使得图像中的轮廓变为白色
# cv2.findContours：用于查找图像中的轮廓。返回两个值：img_contours 包含检测到的轮廓，img_hierarchy 包含轮廓的层次结构信息
img_contours, img_hierarchy = cv2.findContours(img_threshold.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 打印检测到的轮廓信息
print(f"img contours: {img_contours}")
print(f"img img_hierarchy: {img_hierarchy}")

# 创建一个与阈值处理后的图像相同大小的黑色图像
img_new = np.zeros(img_threshold.shape, np.uint8)

# cv2.drawContours 在图像上绘制轮廓，将找到的轮廓信息画用指定颜色出来
cv2.drawContours(img_new, img_contours, -1, (255, 255, 255), 1)
# 创建一个名为 "img_contours" 的窗口，并使用 cv2.WINDOW_NORMAL 设置窗口属性为可调整大小
cv2.namedWindow("img_contours", cv2.WINDOW_NORMAL)
# 在 "img_contours" 窗口中显示绘制了轮廓的图像
cv2.imshow('img_contours', img_new)


# 遍历检测到的轮廓列表，在区域周围放置矩形
for c in img_contours:
    # 使用 cv2.boundingRect 函数计算轮廓的垂直边界最小矩形，得到矩形的左上角坐标 (x, y) 和矩形的宽度 w、高度 h
    (x, y, w, h) = cv2.boundingRect(c)
    # 使用 cv2.rectangle 函数在原始图像 img1 上画出垂直边界最小矩形，矩形的颜色为绿色 (0, 255, 0)，线宽度为2
    cv2.rectangle(img1, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # 使用 cv2.rectangle 函数在原始图像 img2 上画出垂直边界最小矩形，矩形的颜色为绿色 (0, 255, 0)，线宽度为2
    cv2.rectangle(img2, (x, y), (x + w, y + h), (0, 255, 0), 2)


time_end = time.time()
print(f"耗时：{time_end - time_start}")

# 用cv2.imshow 展现最终对比之后的图片
cv2.namedWindow("SSIM Before", cv2.WINDOW_NORMAL)
cv2.imshow('SSIM Before', img1)
cv2.namedWindow("SSIM After", cv2.WINDOW_NORMAL)
cv2.imshow('SSIM After', img2)

# 等待用户按下任意键
cv2.waitKey(0)
# 关闭所有打开的窗口
cv2.destroyAllWindows()


# (score, diff) = structural_similarity(img1_gray, img2_gray, win_size=101, full=True)