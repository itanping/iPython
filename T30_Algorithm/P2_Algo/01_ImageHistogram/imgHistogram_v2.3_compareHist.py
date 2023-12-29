"""
以图搜图：图像直方图（Image Histogram）查找相似图像的原理与实现
测试环境：win10 | python 3.9.13 | OpenCV 4.4.0
实验时间：2023-10-27
实例名称：imgHistogram_v2.3_compareHist.py
"""

# ---------------------------------------------------------------------------------------------------------------------
# 两图测试
# ---------------------------------------------------------------------------------------------------------------------

import cv2
import os

def get_degreeHist(img1_hist, img2_hist):
    # 计算直方图的重合度
    degree = 0
    for i in range(len(img1_hist)):
        if img1_hist[i] != img2_hist[i]:
            degree = degree + (1 - abs(img1_hist[i] - img2_hist[i]) / max(img1_hist[i], img2_hist[i]))
        else:
            degree = degree + 1
    degree = degree / len(img1_hist)
    return degree


def all_compareHist(img1_path, img2_path):
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    img1_hist = cv2.calcHist([img1], [0], None, [256], [0, 256])
    img2_hist = cv2.calcHist([img2], [0], None, [256], [0, 256])

    similarity = get_degreeHist(img1_hist, img2_hist)
    print(f"图像 {os.path.basename(img2_path)} 与目标图像 {os.path.basename(img1_path)} 的相似度：", similarity)

    # 计算直方图相似度
    # cv2.HISTCMP_BHATTACHARYYA: 巴氏距离比较，值越接近 0 表示颜色分布越相似
    similarity = cv2.compareHist(img1_hist, img2_hist, cv2.HISTCMP_BHATTACHARYYA)
    print(f"图像 {os.path.basename(img2_path)} 与目标图像 {os.path.basename(img1_path)} 的相似度（HISTCMP_BHATTACHARYYA/巴氏距离）：", similarity)

    # cv2.HISTCMP_CHISQR: 卡方比较，值越接近 0 表示颜色分布越相似
    similarity = cv2.compareHist(img1_hist, img2_hist, cv2.HISTCMP_CHISQR)
    print(f"图像 {os.path.basename(img2_path)} 与目标图像 {os.path.basename(img1_path)} 的相似度（HISTCMP_CHISQR/卡方比较）：", similarity)

    # cv2.HISTCMP_CORREL: 相关性比较，值越接近 1 表示颜色分布越相似
    similarity = cv2.compareHist(img1_hist, img2_hist, cv2.HISTCMP_CORREL)
    print(f"图像 {os.path.basename(img2_path)} 与目标图像 {os.path.basename(img1_path)} 的相似度（HISTCMP_CORREL/相关性）：", similarity)

    # cv2.HISTCMP_INTERSECT: 直方图交集比较，值越大表示颜色分布越相似
    similarity = cv2.compareHist(img1_hist, img2_hist, cv2.HISTCMP_INTERSECT)
    print(f"图像 {os.path.basename(img2_path)} 与目标图像 {os.path.basename(img1_path)} 的相似度（HISTCMP_INTERSECT/交集比较）：", similarity)
    return similarity

# 目标图像素材库文件夹路径
database_dir = '../../P0_Doc/img_data/'
# 文件路径
img1_path = database_dir + 'car-101.jpg'
img2_path = database_dir + 'car-102.jpg'
img2_path = database_dir + 'car-103.jpg'

all_compareHist(img1_path, img2_path)