"""
以图搜图：余弦相似度（Cosine Similarity）查找相似文本的原理与实现
实验目的：文件读取和预处理
实验环境：Win10 | python 3.9.13 | OpenCV 4.4.0 | numpy 1.21.1 | Matplotlib 3.7.1 | jieba 0.42.1
实验时间：2023-11-30
实例名称：txtConsineSimilarity_v1.0.py
"""

import re
import jieba

# 预处理目标文本
def preprocess_text(text):
    print(f"文本文件原内容：{text}")
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

    # 读取目标文本
    with open(origin_file, 'r', encoding='utf-8') as file:
        origin_text = file.read()

    # 预处理目标文本
    origin_context = preprocess_text(origin_text)
