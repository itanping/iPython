"""
以图搜图：结构相似性（Structural Similarity，简称SSIM算法）查找相似图像的原理与实现
实验环境：Win10 | python 3.9.13 | OpenCV 4.4.0 | numpy 1.21.1 | Matplotlib 3.7.1
实验时间：2024-01-23
实例名称：SSIM_v1.2_inline.py
"""

import cv2
import time
import numpy as np
from skimage.metrics import structural_similarity

time_start = time.time()

# 目标图像素材库文件夹路径
database_dir = '../../P0_Doc/img_data/'

# 读取查询图像和数据库中的图像
img1_path = database_dir + 'iphone15-001.jpg'
img2_path = database_dir + 'iphone15-002.jpg'
img1_path = database_dir + 'car-101.jpg'
img2_path = database_dir + 'car-102.jpg'
img1_path = database_dir + 'apple-101.jpg'
img2_path = database_dir + 'apple-102.jpg'

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
# print(f"两个灰度图像之间的相似性指数：{score}")
# print(f"两个灰度图像之间的图像结构差异：\n{diff_img}")

# 将差异图像的像素值缩放到 [0, 255] 范围，并转换数据类型为 uint8，以便显示
diff_img = (diff_img * 255).astype("uint8")

# 将图像水平拼接在一起（图像1、图像2、差异图像）
result_img = np.hstack((img1_gray, img2_gray, diff_img))

# 创建一个名为 "img comparison" 的窗口，并使用 cv2.WINDOW_NORMAL 设置窗口属性为可调整大小
cv2.namedWindow("Comparison", cv2.WINDOW_NORMAL)
# 在 "img comparison" 窗口中显示拼接后的图像
cv2.imshow("Comparison", result_img)

# 等待用户按下任意键
cv2.waitKey(0)
# 关闭所有打开的窗口
cv2.destroyAllWindows()

time_end = time.time()
print(f"耗时：{time_end - time_start}")