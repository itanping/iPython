"""
以图搜图：余弦相似度（Cosine Similarity）查找相似图像的原理与实现
实验环境：Win10 | python 3.9.13 | OpenCV 4.4.0 | numpy 1.21.1 | Matplotlib 3.7.1
实验时间：2023-11-30
实例名称：imgCosineSimilarity_v1.2.py
"""

import os
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt

def extract_features(image_path):
    # 读取图像并将其转换为灰度
    image = cv2.imread(image_path, cv2.COLOR_BGR2GRAY)
    
    # 计算直方图
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    
    # 归一化直方图
    # cv2.normalize(hist, hist): 这一步是将直方图进行归一化，确保其数值范围在 [0, 1] 之间。归一化是为了消除图像的大小或强度的差异，使得直方图更具有通用性
    # .flatten(): 这一步将归一化后的直方图展平成一维数组。在余弦相似度计算中，我们需要将特征表示成一维向量，以便进行向量之间的相似度比较
    hist = cv2.normalize(hist, hist).flatten()
    return hist

def cosine_similarity(vector1, vector2):
    # 算向量 vector1 和 vector2 的点积，即对应元素相乘后相加得到的标量值
    dot_product = np.dot(vector1, vector2)
    # 计算向量 vector1 的 L2 范数，即向量各元素平方和的平方根
    norm_vector1 = np.linalg.norm(vector1)
    # 计算向量 vector2 的 L2 范数
    norm_vector2 = np.linalg.norm(vector2)
    # 利用余弦相似度公式计算相似度，即两个向量的点积除以它们的 L2 范数之积
    similarity = dot_product / (norm_vector1 * norm_vector2)
    return similarity

def image_search(query_path, database_paths):
    # 提取查询图像的特征
    query_feature = extract_features(query_path)
    
    # 遍历数据库图像并比较相似度
    similaritys = []
    for database_path in database_paths:
        # 提取数据库图像的特征
        database_feature = extract_features(database_path)
        # 计算余弦相似度
        similarity = cosine_similarity(query_feature, database_feature)
        # 将结果保存到列表中（仅保留相似值大于等于 0.8 的图像）
        if (similarity >= 0.65):
            similaritys.append((database_path, similarity))
    
    # 按相似度降序排序
    similaritys.sort(key=lambda x: x[1], reverse=True)
    return similaritys

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

if __name__ == "__main__":
    time_start = time.time()

    # 目标图像素材库文件夹路径
    database_folder_path = '../../P0_Doc/img_data/'
    # 指定测试图像文件扩展名
    img_suffix = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']

    # 目标查询图像路径
    query_image_path = database_folder_path + 'apple-101.jpg'
    query_image_path = database_folder_path + 'X3-01.jpg'
    query_image_path = database_folder_path + 'Q3-01.jpg'
    query_image_path = database_folder_path + 'car-101.jpg'
    
    # 获取目标图像素材库文件夹中所有图像的路径
    all_files = [os.path.join(database_folder_path, filename) for filename in os.listdir(database_folder_path)]

    # 筛选出指定后缀的图像文件
    img_files = [file for file in all_files if any(file.endswith(suffix) for suffix in img_suffix)]
    
    # 进行相似图像搜索
    search_results = image_search(query_image_path, img_files)
    
    # 打印结果
    for similarity in search_results:
        print(f"图像名称：{similarity[0]}，与目标图像 {os.path.basename(query_image_path)} 的近似值：{similarity[1]}")

    time_end = time.time()
    print(f"耗时：{time_end - time_start}")

    # 显示目标相似图像
    show_similar_images(search_results)
