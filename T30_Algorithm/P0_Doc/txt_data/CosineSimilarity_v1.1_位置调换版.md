"""
以图搜图：余弦相似度（Cosine Similarity）查找相似文本的原理与实现
实验目的：比较2个文件相似性，可视化词频相似向量
实验环境：Win10 | python 3.9.13 | OpenCV 4.4.0 | numpy 1.21.1
实验时间：2023-11-30
"""

import re
import numpy as np
import jieba
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from sklearn.feature_extraction.text import CountVectorizer

# 获取文件余弦相似度
def get_cosine_similarity(origin_file, target_file):
    # 读取原始文本
    with open(origin_file, 'r', encoding='utf-8') as file:
        origin_text = file.read()
    # 预处理原始文本
    origin_context = preprocess_text(origin_text)
    print(f"预处理原始文本：{origin_context}")

    # 构建文本向量：使用词袋模型表示文本，过滤停用词
    origin_vectorizer = CountVectorizer(stop_words='english')
    # 使用 CountVectorizer 将原始文本 origin_context 转换为词袋模型的向量表示
    origin_vector = origin_vectorizer.fit_transform([origin_context])
    print(f"原文件词频矩阵：\n{origin_vector}")
    # 转置矩阵，确保维度相同
    origin_vector = origin_vector.T
    # 获取特征单词列表
    feature_names = origin_vectorizer.get_feature_names_out()
    print(f"原文件特征单词列表：{feature_names}")
    print(f"原文件词频向量：\n{origin_vector.toarray()}")

    with open(target_file, 'r', encoding='utf-8') as file:
        target_text = file.read()
        target_context = preprocess_text(target_text)
        print(f"预处理目标文本：{target_context}")

    # 构建文本向量：使用词袋模型表示文本，过滤停用词，并确保与查找源的向量维度一致
    target_vectorizer = CountVectorizer(stop_words='english', vocabulary=feature_names)

    target_vector = target_vectorizer.fit_transform([target_context])
    print(f"目标文件词频矩阵：\n{target_vector}")

    # 转置矩阵，确保维度相同
    target_vector = target_vector.T
    print(f"目标文件转置矩阵：\n{target_vector}")
    print(f"目标文件词频向量：\n{target_vector.toarray()}")

    # 计算余弦相似度
    similarity = cosine_similarity(origin_vector.toarray(), target_vector.toarray())
    print(f"文件 {target_file}，与原文件 {origin_file} 的相似值：{similarity}")

    # 可视化文本向量
    show_text_vectors(origin_vector.toarray(), target_vector.toarray(), feature_names)

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

def show_text_vectors(origin_vector, target_vector, feature_names):
    # 设置中文字体
    font = FontProperties(fname="../../P0_Doc/fonts/msyh.ttc", size=12)
    plt.figure(figsize=(10, 5))
    plt.plot(feature_names, origin_vector, label='Original Text Vector')
    plt.plot(feature_names, target_vector, label='Target Text Vector')
    plt.title('Text Vector Comparison', fontproperties=font)
    plt.xlabel('Feature Names', fontproperties=font)
    plt.ylabel('Vector Values', fontproperties=font)
    plt.xticks(rotation=90, fontproperties=font)
    plt.legend()
    plt.tight_layout()
    plt.show()

# 预处理目标文本
def preprocess_text(text):
    print(f"文本文件内容：{text}")
    # 将文本转换为小写
    text = text.lower()
    print(f"将文本转为小写：{text}")
    # 移除标点符号、数字和中文标点符号
    text = re.sub(r'[^a-z\u4e00-\u9fa5\s]', '', text)
    print(f"移除标点符号后：{text}")
    # 使用 jieba 进行中文分词
    text_words = jieba.cut(text)
    # 将分词结果拼接成字符串
    processed_text = ' '.join(text_words)
    print(f"将分词结果拼接成字符串：{processed_text}")
    return processed_text
    
if __name__ == "__main__":
    # 本地测试文本素材库
    test_dir_path = '../../P0_Doc/txt_data/'
    # 本地测试文本素材路径
    origin_file = test_dir_path + 'CosineSimilarity_定义_org.txt'
    target_file = test_dir_path + 'CosineSimilarity_定义_v1.0.txt'

    # 获取文件余弦相似度
    get_cosine_similarity(origin_file, target_file)
