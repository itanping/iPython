"""
以图搜图：均值哈希算法（Average Hash Algorithm，简称aHash）的原理与实现
测试环境：win10 | python 3.9.13 | OpenCV 4.4.0 | numpy 1.21.1
实验时间：2023-10-20
"""

# ---------------------------------------------------------------------------------------------------------------------
# 多图测试
# ---------------------------------------------------------------------------------------------------------------------

import os
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt

def get_hash(img_path):
    # 读取图片：通过OpenCV加载图像
    img = cv2.imread(img_path)

    # 通道重排，从BGR转换为RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 缩小图像：使用OpenCV的resize函数将图像缩放为8x8像素，采用Cubic插值方法进行图像重采样（目的是确保图像的一致性，并降低计算的复杂度）
    # cv2.INTER_NEAREST：最近邻插值，也称为最近邻算法。它简单地使用最接近目标像素的原始像素的值。虽然计算速度快，但可能导致图像质量下降。
    # cv2.INTER_LINEAR：双线性插值，通过对最近的4个像素进行线性加权来估计目标像素的值。比最近邻插值更精确，但计算成本略高。
    # cv2.INTER_CUBIC：双三次插值，使用16个最近像素的加权平均值来估计目标像素的值。通常情况下，这是一个不错的插值方法，适用于图像缩小。
    # cv2.INTER_LANCZOS4：Lanczos插值，一种高质量的插值方法，使用Lanczos窗口函数。通常用于缩小图像，以保留图像中的细节和纹理。
    img_resize = cv2.resize(img, (8, 8), cv2.INTER_CUBIC)

    # 图像灰度化：将彩色图像转换为灰度图像。
    img_gray = cv2.cvtColor(img_resize, cv2.COLOR_BGR2GRAY)
    # print(f"缩放8x8的图像中每个像素的颜色=\n{img_gray}")

    img_average = np.mean(img_gray) 
    # print(f"灰度图像中所有像素的平均值={img_average}")

    # 遍历图像像素：嵌套循环遍历图像的所有像素，对比灰度图像的平均灰度值，转换为二进制的图像哈希值
    # img_gray：是灰度图像
    # img_gray.shape[0] 和 img_gray.shape[1] 分别表示图像的高度和宽度
    img_hash_binary = [] 
    for i in range(img_gray.shape[0]): 
        for j in range(img_gray.shape[1]): 
            if img_gray[i,j] >= img_average: 
                img_hash_binary.append(1)
            else: 
                img_hash_binary.append(0)
    # print(f"对比灰度图像的平均像素值降噪（图像的二进制哈希值）数组={img_hash_binary}")

    # 将列表中的元素转换为字符串并连接起来，形成一组64位的图像二进制哈希值字符串
    img_hash_binary_str = ''.join(map(str, img_hash_binary))
    # print(f"对比灰度图像的平均像素值降噪（图像的二进制哈希值）={img_hash_binary_str}")

    # # lambda表达式
    # img_hash_binary_str = ""
    # for i in range(8):
    #     img_hash_binary_str += ''.join(map(lambda i: '0' if i < img_average else '1', img_gray[i]))
    # print(f"对比灰度图像的平均像素值降噪（图像的二进制哈希值）={img_hash_binary_str}")

    # 用于存储图像哈希值的十六进制
    img_hash = ""
    # 遍历二进制哈希值：通过循环，代码以4位为一组遍历二进制哈希值 img_hash_binary_str。
    # range(0, 64, 4) 确保代码在哈希值的每4位之间进行迭代。
    for i in range(0, 64, 4):
        # 将4位二进制转换为一个十六进制字符
        # 在每次循环中，代码取出哈希值中的4位二进制（例如，img_hash_binary_str[i : i + 4]）
        # 然后使用'%x' % int(..., 2)将这4位二进制转换为一个十六进制字符。
        # int(..., 2)将二进制字符串转换为整数，'%x'将整数转换为十六进制字符。
        # 将十六进制字符追加到 img_hash：在每次循环中，得到的十六进制字符将被追加到 img_hash 字符串中。
        img_hash += "".join('%x' % int(img_hash_binary_str[i : i + 4], 2))
    # print(f"图像可识别的哈希值={img_hash}")
    return img_hash


# 汉明距离：计算两个等长字符串（通常是二进制字符串或位字符串）之间的汉明距离。用于确定两个等长字符串在相同位置上不同字符的数量。
def hamming_distance(s1, s2):
    # 检查这两个字符串的长度是否相同。如果长度不同，它会引发 ValueError 异常，因为汉明距离只适用于等长的字符串
    if len(s1) != len(s2):
        raise ValueError("Input strings must have the same length")
    
    distance = 0
    for i in range(len(s1)):
        # 遍历两个字符串的每个字符，比较它们在相同位置上的值。如果发现不同的字符，将 distance 的值增加 1
        if s1[i] != s2[i]:
            distance += 1
    return distance


# ---------------------------------------- 测试 ----------------------------------------

time_start = time.time()

script_dir = os.path.dirname(__file__)
print(script_dir)

# 测试图片全路径
img_1 = os.path.join(script_dir, 'img_test/apple-01.jpg')
img_2 = os.path.join(script_dir, 'img_test/apple-02.jpg')
img_3 = os.path.join(script_dir, 'img_test/apple-03.jpg')
img_4 = os.path.join(script_dir, 'img_test/apple-04.jpg')
img_5 = os.path.join(script_dir, 'img_test/apple-05.jpg')
img_6 = os.path.join(script_dir, 'img_test/apple-06.jpg')
img_7 = os.path.join(script_dir, 'img_test/apple-07.jpg')
img_8 = os.path.join(script_dir, 'img_test/apple-08.jpg')
img_9 = os.path.join(script_dir, 'img_test/apple-09.jpg')
img_10 = os.path.join(script_dir, 'img_test/pear-001.jpg')

img_hash1 = get_hash(img_1)
img_hash2 = get_hash(img_2)
img_hash3 = get_hash(img_3)
img_hash4 = get_hash(img_4)
img_hash5 = get_hash(img_5)
img_hash6 = get_hash(img_6)
img_hash7 = get_hash(img_7)
img_hash8 = get_hash(img_8)
img_hash9 = get_hash(img_9)
img_hash10 = get_hash(img_10)

distance1 = hamming_distance(img_hash1, img_hash1)
distance2 = hamming_distance(img_hash1, img_hash2)
distance3 = hamming_distance(img_hash1, img_hash3)
distance4 = hamming_distance(img_hash1, img_hash4)
distance5 = hamming_distance(img_hash1, img_hash5)
distance6 = hamming_distance(img_hash1, img_hash6)
distance7 = hamming_distance(img_hash1, img_hash7)
distance8 = hamming_distance(img_hash1, img_hash8)
distance9 = hamming_distance(img_hash1, img_hash9)
distance10 = hamming_distance(img_hash1, img_hash10)

time_end = time.time()

print(f"图片名称：{img_1}，图片HASH：{img_hash1}，与图片1的近似值（汉明距离）：{distance1}")
print(f"图片名称：{img_2}，图片HASH：{img_hash2}，与图片1的近似值（汉明距离）：{distance2}")
print(f"图片名称：{img_3}，图片HASH：{img_hash3}，与图片1的近似值（汉明距离）：{distance3}")
print(f"图片名称：{img_4}，图片HASH：{img_hash4}，与图片1的近似值（汉明距离）：{distance4}")
print(f"图片名称：{img_5}，图片HASH：{img_hash5}，与图片1的近似值（汉明距离）：{distance5}")
print(f"图片名称：{img_6}，图片HASH：{img_hash6}，与图片1的近似值（汉明距离）：{distance6}")
print(f"图片名称：{img_7}，图片HASH：{img_hash7}，与图片1的近似值（汉明距离）：{distance7}")
print(f"图片名称：{img_8}，图片HASH：{img_hash8}，与图片1的近似值（汉明距离）：{distance8}")
print(f"图片名称：{img_9}，图片HASH：{img_hash9}，与图片1的近似值（汉明距离）：{distance9}")
print(f"图片名称：{img_10}，图片HASH：{img_hash10}，与图片1的近似值（汉明距离）：{distance10}")

print(f"耗时：{time_end - time_start}")