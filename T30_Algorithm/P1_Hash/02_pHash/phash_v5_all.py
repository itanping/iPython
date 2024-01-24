"""
以图搜图：感知哈希算法（Perceptual Hash Algorithm，简称pHash）的原理与实现
测试环境：win10 | python 3.9.13 | OpenCV 4.4.0 | numpy 1.21.1
实验时间：2023-10-31
"""

import os
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt

def get_pHash(img_path):
    # 读取图像：通过OpenCV的imread加载RGB图像
    img_rgb = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    # 缩小图像：使用OpenCV的resize函数将图像缩放为32x32像素，采用Cubic插值方法进行图像重采样
    img_resize = cv2.resize(img_rgb, (32, 32), cv2.INTER_CUBIC)
    # 图像灰度化：将彩色图像转换为灰度图像
    img_gray = cv2.cvtColor(img_resize, cv2.COLOR_BGR2GRAY)
    # print(f"缩放32x32的图像中每个像素的颜色=\n{img_gray}")

    # 离散余弦变换（DCT）：计算图像的DCT变换，得到32×32的DCT变换系数矩阵
    img_dct = cv2.dct(np.float32(img_gray))
    # print(f"灰度图像离散余弦变换（DCT）={img_dct}")

    # 缩放DCT：将DCT系数的大小显式地调整为8x8。然后它计算调整后的DCT系数的均值，并生成哈希值。
    img_dct.resize(8, 8)

    # 计算灰度均值：计算DCT变换后图像块的均值
    img_avg = np.mean(img_dct)
    # print(f"DCT变换后图像块的均值={img_avg}")

    img_hash_str = ""
    for i in range(8):
        img_hash_str += ''.join(map(lambda i: '0' if i < img_avg else '1', img_dct[i]))
    # print(f"图像的二进制哈希值={img_hash_str}")

    # 生成图像可识别哈希值
    img_hash = ''.join(map(lambda x:'%x' % int(img_hash_str[x : x + 4], 2), range(0, 64, 4)))
    # print(f"图像可识别的哈希值={img_hash}")

    """
    # # 版本二
    # # 生成二进制哈希值
    # img_hash_str = ''
    # for i in range(8):
    #     for j in range(8):
    #         if img_dct[i, j] > img_avg:
    #             img_hash_str += '1'
    #         else:
    #             img_hash_str += '0'
    # print(f"图像的二进制哈希值={img_hash_str}")

    # # 生成图像可识别哈希值
    # img_hash = ''
    # for i in range(0, 64, 4):
    #     img_hash += ''.join('%x' % int(img_hash_str[i: i + 4], 2))
    # print(f"图像可识别的哈希值={img_hash}")
    """

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


# ------------------------------------------------- 测试 -------------------------------------------------
time_start = time.time()

# 指定测试图像库目录
img_dir = 'img_test'
# 指定测试图像文件扩展名
img_suffix = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']

# 获取当前执行脚本所在目录
script_dir = os.path.dirname(__file__)
# 获取目标测试图像的全路径
img_org_path = os.path.join(script_dir, img_dir, 'apple-01.jpg')
# 获取目标图像可识别哈希值（图像指纹）
org_img_hash = get_pHash(img_org_path)
print(f"目标图像：{os.path.relpath(img_org_path)}，图片HASH：{org_img_hash}")

# 获取测试图像库中所有文件
all_files = os.listdir(os.path.join(script_dir, img_dir))
# 筛选出指定后缀的图像文件
img_files = [file for file in all_files if any(file.endswith(suffix) for suffix in img_suffix)]

img_hash_all = []
# 遍历测试图像库中的每张图像
for img_file in img_files:
    # 获取相似图像文件路径
    img_path = os.path.join(script_dir, img_dir, img_file)
    # 获取相似图像可识别哈希值（图像指纹）
    img_hash = get_pHash(img_path)
    # 获取相似图像与目标图像的汉明距离
    distance = hamming_distance(org_img_hash, img_hash)
    # 存储相似图像的相对路径、哈希值、汉明距离
    img_hash_all.append((os.path.relpath(img_path), img_hash, distance))

for img in img_hash_all:
    print(f"图像：{img[0]}，图像HASH：{img[1]}，与目标图像的相似值（汉明距离）：{img[2]}")

time_end = time.time()
print(f"耗时：{time_end - time_start}")