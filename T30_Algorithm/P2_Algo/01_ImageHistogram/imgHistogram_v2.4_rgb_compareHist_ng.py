"""
以图搜图：图像直方图（Image Histogram）查找相似图像的原理与实现
测试环境：win10 | python 3.9.13 | OpenCV 4.4.0
实验时间：2023-10-27
实例名称：imgHistogram_v2.4_rgb_compareHist_ng.py
"""

# ---------------------------------------------------------------------------------------------------------------------
# 两图测试：先灰度，再使用RGB。报错！！！
# ---------------------------------------------------------------------------------------------------------------------

import cv2

def calcHist(img1_path, img2_path):
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    # 将图像转换为灰度图像
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    img1_hist = cv2.calcHist([img1_gray], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    img2_hist = cv2.calcHist([img2_gray], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    similarity = cv2.compareHist(img1_hist, img2_hist, cv2.HISTCMP_BHATTACHARYYA)
    print(f"图像 {img2_path} 与目标图像 {img1_path} 的相似度（HISTCMP_BHATTACHARYYA/巴氏距离）：", similarity)
    return similarity

# 目标图像素材库文件夹路径
database_dir = '../../P0_Doc/img_data/'
# 文件路径
img1_path = database_dir + 'car-101.jpg'
img2_path = database_dir + 'car-102.jpg'

calcHist(img1_path, img2_path)