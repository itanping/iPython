"""
以图搜图：图像直方图（Image Histogram）查找相似图像的原理与实现
测试环境：win10 | python 3.9.13 | OpenCV 4.4.0 | numpy 1.21.1 | matplotlib 3.7.1
实验时间：2023-10-27
实例名称：imgHistogram_v3.1_rgb.py
"""

# ---------------------------------------------------------------------------------------------------------------------
# 多图测试
# ---------------------------------------------------------------------------------------------------------------------

import os
import time
import cv2

def get_calcHist(org_img_hist, img_path):
    # 读取图像：通过OpenCV的imread加载RGB图像
    img = cv2.imread(img_path)
    img_hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])

    # 计算直方图相似度
    # cv2.HISTCMP_CORREL: 相关性比较，值越接近 1 表示颜色分布越相似
    # cv2.HISTCMP_CHISQR: 卡方比较，值越接近 0 表示颜色分布越相似
    # cv2.HISTCMP_BHATTACHARYYA: 巴氏距离比较，值越接近 0 表示颜色分布越相似
    # cv2.HISTCMP_INTERSECT: 直方图交集比较，值越大表示颜色分布越相似
    similarity = cv2.compareHist(org_img_hist, img_hist, cv2.HISTCMP_BHATTACHARYYA)
    # print(similarity)
    return similarity


# ------------------------------------------------ 测试 ------------------------------------------------
if __name__ == '__main__':
    time_start = time.time()

    # 指定测试图像库目录
    img_dir = '../../P0_Doc/img_data/'
    # 指定测试图像文件扩展名
    img_suffix = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']

    # 获取当前执行脚本所在目录
    script_dir = os.path.dirname(__file__)
    # 获取目标测试图像的全路径
    img_org_path = os.path.join(script_dir, img_dir, 'apple-101.jpg')
    # 加载要查询的图像
    query_image = cv2.imread(img_org_path)

    # 计算查询图像的直方图：灰度直方图算法
    img_org_hist = cv2.calcHist([query_image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    print(f"目标图像：{os.path.relpath(img_org_path)}")

    # 获取测试图像库中所有文件
    all_files = os.listdir(os.path.join(script_dir, img_dir))
    # 筛选出指定后缀的图像文件
    img_files = [file for file in all_files if any(file.endswith(suffix) for suffix in img_suffix)]

    # 存储相似度值和对应的图像路径
    similarities = []
    # 遍历测试图像库中的每张图像
    for img_file in img_files:
        # 获取相似图像文件路径
        img_path = os.path.join(script_dir, img_dir, img_file)
        # 获取相似图像可识别哈希值（图像指纹）
        similarity = get_calcHist(img_org_hist, img_path)
        # 存储相似度值和对应的图像路径
        similarities.append((os.path.relpath(img_path), similarity))

    for similarity in similarities:
        if (similarity[1] <= 0.5):
            print(f"图像名称：{similarity[0]}，与目标图像 {os.path.basename(img_org_path)} 的近似值：{similarity[1]}")

    time_end = time.time()
    print(f"耗时：{time_end - time_start}")