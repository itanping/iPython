"""
以图搜图：余弦相似度（Cosine Similarity）查找相似图像的原理与实现
实验环境：Win10 | python 3.9.13 | OpenCV 4.4.0 | numpy 1.21.1 | Matplotlib 3.7.1
实验时间：2023-11-30
实例名称：imgCosineSimilarity_v1.0.py
"""

import cv2
import numpy as np

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

# 计算图像的直方图
img1_hist = cv2.calcHist([img1_gray], [0], None, [256], [0, 256])
img2_hist = cv2.calcHist([img2_gray], [0], None, [256], [0, 256])

# 计算余弦相似度
similarity = np.dot(img1_hist.flatten(), img2_hist.flatten()) / (np.linalg.norm(img1_hist) * np.linalg.norm(img2_hist))
print(f"相似度：{similarity}")

# 归一化直方图：将特征表示成一维向量
vector1 = img1_hist.flatten()
vector2 = img2_hist.flatten()
# 计算向量 vector1 和 vector2 的点积，即对应元素相乘后相加得到的标量值
dot_product = np.dot(vector1, vector2)
# 计算向量 vector1 的 L2 范数，即向量各元素平方和的平方根
norm_vector1 = np.linalg.norm(vector1)
# 计算向量 vector2 的 L2 范数
norm_vector2 = np.linalg.norm(vector2)
# 利用余弦相似度公式计算相似度，即两个向量的点积除以它们的 L2 范数之积
similarity = dot_product / (norm_vector1 * norm_vector2)
print(f"图像名称：{img2_path}，与目标图像 {img1_path} 的近似值：{similarity}")

if (similarity > 0.8):
    print(f"图像名称：{img2_path}，与目标图像 {img1_path} 的近似值：{similarity}")
