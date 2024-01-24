"""
以图搜图：图像直方图（Image Histogram）查找相似图像的原理与实现
测试环境：win10 | python 3.9.13 | OpenCV 4.4.0 | numpy 1.21.1 | matplotlib 3.7.1
实验场景：图像测试素材库中，查找所有图像，找出与目标图像相似值小于等于0.7的图像
实验时间：2023-10-27
实例名称：imgHistogram_v3.2_gray_show.py
"""

import os
import time
import cv2
import matplotlib.pyplot as plt

def get_calcHist(org_img_hist, img_path):
    # 读取图像：通过OpenCV的imread加载RGB图像
    img = cv2.imread(img_path)
    # img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
    img_hist = cv2.calcHist([img], [0], None, [256], [0, 256])

    # 计算直方图相似度
    # cv2.HISTCMP_CORREL: 相关性比较，值越接近 1 表示颜色分布越相似
    # cv2.HISTCMP_CHISQR: 卡方比较，值越接近 0 表示颜色分布越相似
    # cv2.HISTCMP_BHATTACHARYYA: 巴氏距离比较，值越接近 0 表示颜色分布越相似
    # cv2.HISTCMP_INTERSECT: 直方图交集比较，值越大表示颜色分布越相似
    similarity = cv2.compareHist(org_img_hist, img_hist, cv2.HISTCMP_BHATTACHARYYA)
    return similarity

def show_similar_images(similar_images, images_per_column=3):
    # 计算总共的图片数量
    num_images = len(similar_images)
    # 计算所需的行数
    num_rows = (num_images + images_per_column - 1) // images_per_column

    # 创建一个子图，每行显示 images_per_column 张图片
    fig, axes = plt.subplots(num_rows, images_per_column, figsize=(12, 15), squeeze=False)
    
    # 遍历每一行
    for row in range(num_rows):
        # 遍历每一列
        for col in range(images_per_column):
            # 计算当前图片在列表中的索引
            index = row * images_per_column + col
            # 检查索引是否越界
            if index < num_images:
                # 获取当前相似图片的路径和相似度
                image_path = similar_images[index][0]
                similarity = similar_images[index][1]
                
                # 读取图片并转换颜色通道
                image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

                # 在子图中显示图片
                axes[row, col].imshow(image)
                # 设置子图标题，包括图片路径和相似度
                axes[row, col].set_title(f"Similar Image: {os.path.basename(image_path)} \n Similar Score: {similarity:.4f}")
                # 关闭坐标轴
                axes[row, col].axis('off')
    # 显示整个图
    plt.show()

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
    # query_image = cv2.cvtColor(cv2.imread(img_org_path), cv2.COLOR_BGR2GRAY)

    # 计算查询图像的直方图：灰度直方图算法
    img_org_hist = cv2.calcHist([query_image], [0], None, [256], [0, 256])
    print(f"目标图像：{os.path.relpath(img_org_path)}")

    # 获取测试图像库中所有文件
    all_files = os.listdir(os.path.join(script_dir, img_dir))
    # 筛选出指定后缀的图像文件
    img_files = [file for file in all_files if any(file.endswith(suffix) for suffix in img_suffix)]

    # 存储相似度值和对应的图像路径
    img_search_results = []
    # 遍历测试图像库中的每张图像
    for img_file in img_files:
        # 获取相似图像文件路径
        img_path = os.path.join(script_dir, img_dir, img_file)
        # 获取相似图像可识别哈希值（图像指纹）
        similarity = get_calcHist(img_org_hist, img_path)
        # print(f"图像名称：{img_path}，与目标图像 {os.path.basename(img_org_path)} 的近似值：{similarity}")

        if (similarity <= 0.7):
            # 存储相似度值和对应的图像路径
            img_search_results.append((os.path.relpath(img_path), similarity))

    # 按相似度升序排序
    img_search_results.sort(key=lambda item: item[1])

    for img_similarity in img_search_results:
        print(f"图像名称：{img_similarity[0]}，与目标图像 {os.path.basename(img_org_path)} 的相似值：{img_similarity[1]}")

    time_end = time.time()
    print(f"耗时：{time_end - time_start}")

    # 显示目标相似图像
    show_similar_images(img_search_results)