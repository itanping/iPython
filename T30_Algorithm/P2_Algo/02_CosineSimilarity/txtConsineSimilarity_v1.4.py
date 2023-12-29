"""
以图搜图：余弦相似度（Cosine Similarity）查找相似文本的原理与实现
实验目的：比较2个文件相似性
实验环境：Win10 | python 3.9.13 | OpenCV 4.4.0 | numpy 1.21.1 | Matplotlib 3.7.1 | jieba 0.42.1
实验时间：2023-11-30
实例名称：txtConsineSimilarity_v1.4.py
"""

import os
import re
import time
import numpy as np
import jieba
from sklearn.feature_extraction.text import CountVectorizer

# 预处理目标文本
def preprocess_text(text):
    # 将文本转换为小写
    text = text.lower()
    # 移除标点符号、数字和中文标点符号
    text = re.sub(r'[^a-z\u4e00-\u9fa5\s]', '', text)
    # 使用 jieba 进行中文分词
    text_words = jieba.cut(text)
    # 将分词结果拼接成字符串
    processed_text = ' '.join(text_words)
    return processed_text
    
def cosine_similarity(vector1, vector2):
    # 将二维列向量转换为一维数组
    vector1 = vector1.flatten()
    vector2 = vector2.flatten()
    # 算向量 vector1 和 vector2 的点积，即对应元素相乘后相加得到的标量值
    dot_product = np.dot(vector1, vector2)
    # 计算向量 vector1 的 L2 范数，即向量各元素平方和的平方根
    norm_vector1 = np.linalg.norm(vector1)
    # 计算向量 vector2 的 L2 范数
    norm_vector2 = np.linalg.norm(vector2)
    # 避免除零错误
    if norm_vector1 == 0 or norm_vector2 == 0:
        return 0
    # 利用余弦相似度公式计算相似度，即两个向量的点积除以它们的 L2 范数之积
    similarity = dot_product / (norm_vector1 * norm_vector2)
    return similarity

# 获取文件余弦相似度
def get_cosine_similarity(origin_file, test_files):
    # 读取原始文本
    with open(origin_file, 'r', encoding='utf-8') as file:
        origin_text = file.read()
    # 预处理原始文本
    origin_context = preprocess_text(origin_text)

    # 构建文本向量：使用词袋模型表示文本，过滤停用词
    origin_vectorizer = CountVectorizer(stop_words='english')
    # 使用 CountVectorizer 将原始文本 origin_context 转换为词袋模型的向量表示
    origin_vector = origin_vectorizer.fit_transform([origin_context])
    # 转置矩阵，确保维度相同
    origin_vector = origin_vector.T
    # 获取特征单词列表
    feature_names = origin_vectorizer.get_feature_names_out()

    # 遍历测试库中的文本文件，获取文件余弦相似度
    for filename in test_files:
        with open(filename, 'r', encoding='utf-8') as file:
            target_text = file.read()
            target_context = preprocess_text(target_text)

            # 构建文本向量：使用词袋模型表示文本，过滤停用词，并确保与查找源的向量维度一致
            target_vectorizer = CountVectorizer(stop_words='english', vocabulary=feature_names)
            target_vector = target_vectorizer.fit_transform([target_context])

            # 转置矩阵，确保维度相同
            target_vector = target_vector.T

            # 计算余弦相似度
            similarity = cosine_similarity(origin_vector.toarray(), target_vector.toarray())
            print(f"文件 {os.path.basename(filename)}，与原文件 {os.path.basename(origin_file)} 的相似值：{similarity}")

            # 根据需求设定一个阈值，将相似度大于阈值的文件视为相似文件，并按相似度结果排序，得到相似度最高的文本文件
            if (similarity >= 0.9):
                text_similarities.append((filename, similarity))

if __name__ == "__main__":
    time_start = time.time()

    # 本地测试文本素材库
    test_dir_path = '../../P0_Doc/txt_data/'
    # 本地测试文本素材路径
    origin_file = test_dir_path + 'CosineSimilarity_org.txt'
    # 指定测试文本文件扩展名
    txt_suffix = ['.txt', '.doc', '.md']

    # 获取素材库文件夹中所有文件路径
    all_files = [os.path.join(test_dir_path, filename) for filename in os.listdir(test_dir_path)]

    # 筛选出指定后缀的文件
    test_files = [file for file in all_files if any(file.endswith(suffix) for suffix in txt_suffix)]

    # 获取素材库文件夹中文件余弦相似度
    text_similarities = []
    get_cosine_similarity(origin_file, test_files)

    # 按相似度降序排序
    text_similarities.sort(key=lambda item: item[1], reverse=True)
    print(f"按相似度降序排序：{text_similarities}")

    # 打印相似度最高的文本文件
    print(f"相似度最高的文本文件: {text_similarities[0][0]}, 相似度: {float(text_similarities[0][1]):.4f}")

    time_end = time.time()
    print(f"耗时：{time_end - time_start}")