"""
以图搜图：图像直方图（Image Histogram）查找相似图像的原理与实现
测试环境：win10 | python 3.9.13 | OpenCV 4.4.0 | numpy 1.21.1 | matplotlib 3.7.1
实验时间：2023-10-27
实例名称：imgHistogram_v2.7_bgr.py
"""

# ---------------------------------------------------------------------------------------------------------------------
# 两图测试
# ---------------------------------------------------------------------------------------------------------------------

import cv2

def calcHist_b(img1_path, img2_path):
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    img1_hist = cv2.calcHist([img1], [0], None, [256], [0, 256])
    img2_hist = cv2.calcHist([img2], [0], None, [256], [0, 256])

    # cv2.HISTCMP_CHISQR: 卡方比较，值越接近 0 表示颜色分布越相似
    similarity = cv2.compareHist(img1_hist, img2_hist, cv2.HISTCMP_CHISQR)
    print(f"使用灰度单通道B（[0]），图像 {img2_path} 与目标图像 {img1_path} 的相似度（HISTCMP_CHISQR/卡方比较）：", similarity)

    # cv2.HISTCMP_CORREL: 相关性比较，值越接近 1 表示颜色分布越相似
    similarity = cv2.compareHist(img1_hist, img2_hist, cv2.HISTCMP_CORREL)
    print(f"使用灰度单通道B（[0]），图像 {img2_path} 与目标图像 {img1_path} 的相似度（HISTCMP_CORREL/相关性）：", similarity)

    # cv2.HISTCMP_INTERSECT: 直方图交集比较，值越大表示颜色分布越相似
    similarity = cv2.compareHist(img1_hist, img2_hist, cv2.HISTCMP_INTERSECT)
    print(f"使用灰度单通道B（[0]），图像 {img2_path} 与目标图像 {img1_path} 的相似度（HISTCMP_INTERSECT/交集比较）：", similarity)

    # cv2.HISTCMP_BHATTACHARYYA: 巴氏距离比较，值越接近 0 表示颜色分布越相似
    similarity = cv2.compareHist(img1_hist, img2_hist, cv2.HISTCMP_BHATTACHARYYA)
    print(f"使用灰度单通道B（[0]），图像 {img2_path} 与目标图像 {img1_path} 的相似度（HISTCMP_BHATTACHARYYA/巴氏距离）：", similarity)
    return similarity

def calcHist_g(img1_path, img2_path):
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    img1_hist = cv2.calcHist([img1], [1], None, [256], [0, 256])
    img2_hist = cv2.calcHist([img2], [1], None, [256], [0, 256])

    # cv2.HISTCMP_CHISQR: 卡方比较，值越接近 0 表示颜色分布越相似
    similarity = cv2.compareHist(img1_hist, img2_hist, cv2.HISTCMP_CHISQR)
    print(f"使用灰度单通道G（[1]），图像 {img2_path} 与目标图像 {img1_path} 的相似度（HISTCMP_CHISQR/卡方比较）：", similarity)

    # cv2.HISTCMP_CORREL: 相关性比较，值越接近 1 表示颜色分布越相似
    similarity = cv2.compareHist(img1_hist, img2_hist, cv2.HISTCMP_CORREL)
    print(f"使用灰度单通道G（[1]），图像 {img2_path} 与目标图像 {img1_path} 的相似度（HISTCMP_CORREL/相关性）：", similarity)

    # cv2.HISTCMP_INTERSECT: 直方图交集比较，值越大表示颜色分布越相似
    similarity = cv2.compareHist(img1_hist, img2_hist, cv2.HISTCMP_INTERSECT)
    print(f"使用灰度单通道G（[1]），图像 {img2_path} 与目标图像 {img1_path} 的相似度（HISTCMP_INTERSECT/交集比较）：", similarity)

    # cv2.HISTCMP_BHATTACHARYYA: 巴氏距离比较，值越接近 0 表示颜色分布越相似
    similarity = cv2.compareHist(img1_hist, img2_hist, cv2.HISTCMP_BHATTACHARYYA)
    print(f"使用灰度单通道G（[1]），图像 {img2_path} 与目标图像 {img1_path} 的相似度（HISTCMP_BHATTACHARYYA/巴氏距离）：", similarity)
    return similarity

def calcHist_r(img1_path, img2_path):
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    img1_hist = cv2.calcHist([img1], [2], None, [256], [0, 256])
    img2_hist = cv2.calcHist([img2], [2], None, [256], [0, 256])

    # cv2.HISTCMP_CHISQR: 卡方比较，值越接近 0 表示颜色分布越相似
    similarity = cv2.compareHist(img1_hist, img2_hist, cv2.HISTCMP_CHISQR)
    print(f"使用灰度单通道R（[2]），图像 {img2_path} 与目标图像 {img1_path} 的相似度（HISTCMP_CHISQR/卡方比较）：", similarity)

    # cv2.HISTCMP_CORREL: 相关性比较，值越接近 1 表示颜色分布越相似
    similarity = cv2.compareHist(img1_hist, img2_hist, cv2.HISTCMP_CORREL)
    print(f"使用灰度单通道R（[2]），图像 {img2_path} 与目标图像 {img1_path} 的相似度（HISTCMP_CORREL/相关性）：", similarity)

    # cv2.HISTCMP_INTERSECT: 直方图交集比较，值越大表示颜色分布越相似
    similarity = cv2.compareHist(img1_hist, img2_hist, cv2.HISTCMP_INTERSECT)
    print(f"使用灰度单通道R（[2]），图像 {img2_path} 与目标图像 {img1_path} 的相似度（HISTCMP_INTERSECT/交集比较）：", similarity)

    # cv2.HISTCMP_BHATTACHARYYA: 巴氏距离比较，值越接近 0 表示颜色分布越相似
    similarity = cv2.compareHist(img1_hist, img2_hist, cv2.HISTCMP_BHATTACHARYYA)
    print(f"使用灰度单通道R（[2]），图像 {img2_path} 与目标图像 {img1_path} 的相似度（HISTCMP_BHATTACHARYYA/巴氏距离）：", similarity)
    return similarity

# 目标图像素材库文件夹路径
database_dir = '../../P0_Doc/img_data/'
# 文件路径
img1_path = database_dir + 'car-101.jpg'
img2_path = database_dir + 'car-102.jpg'
img2_path = database_dir + 'car-103.jpg'

calcHist_b(img1_path, img2_path)
calcHist_g(img1_path, img2_path)
calcHist_r(img1_path, img2_path)