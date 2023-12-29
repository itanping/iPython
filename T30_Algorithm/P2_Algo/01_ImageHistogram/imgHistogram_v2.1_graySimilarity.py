"""
以图搜图：图像直方图（Image Histogram）查找相似图像的原理与实现
测试环境：win10 | python 3.9.13 | OpenCV 4.4.0
实验时间：2023-10-27
实例名称：imgHistogram_v2.1_graySimilarity.py
"""

# ---------------------------------------------------------------------------------------------------------------------
# 两图测试
# ---------------------------------------------------------------------------------------------------------------------

import cv2

def get_calcHist(img1_path, img2_path):
    # 读取图像
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    # 计算图像灰度单通道直方图
    img1_hist = cv2.calcHist([img1], [0], None, [256], [0, 256])
    img2_hist = cv2.calcHist([img2], [0], None, [256], [0, 256])

    # 计算直方图相似度
    # cv2.HISTCMP_CORREL: 相关性比较，值越接近 1 表示颜色分布越相似
    similarity = cv2.compareHist(img1_hist, img2_hist, cv2.HISTCMP_CORREL)
    print("图像2与图像1的相似度（HISTCMP_CORREL/相关性）：", similarity)

    # 或者
    # 计算直方图的重合度
    degree = 0
    for i in range(len(img1_hist)):
        if img1_hist[i] != img2_hist[i]:
            degree = degree + (1 - abs(img1_hist[i] - img2_hist[i]) / max(img1_hist[i], img2_hist[i]))
        else:
            degree = degree + 1
    degree = degree / len(img1_hist)
    print("图像2与图像1的重合度：", degree)
    return similarity


# 目标图像素材库文件夹路径
database_dir = '../../P0_Doc/img_data/'
# 文件路径
img1_path = database_dir + 'car-101.jpg'
img2_path = database_dir + 'car-102.jpg'

print("图像1路径：", img1_path)
print("图像2路径：", img2_path)

get_calcHist(img1_path, img2_path)