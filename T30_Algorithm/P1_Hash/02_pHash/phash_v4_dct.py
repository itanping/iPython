"""
以图搜图：感知哈希算法（Perceptual hash algorithm）的原理与实现
测试环境：win10 | python 3.9.13 | OpenCV 4.4.0 | numpy 1.21.1
实验时间：2023-10-22
"""

# ---------------------------------------------------------------------------------------------------------------------
# 测试：为什么要缩放DCT？DCT缩放方式有哪些？不同DCT缩放方式有何不同？不进行DCT缩放效果会怎么样？
# ---------------------------------------------------------------------------------------------------------------------

import cv2
import time
import numpy as np
import matplotlib.pyplot as plt

# DCT变换后：无特征频率区域缩放，使用整个32x32图像块的频率分布，计算整个DCT系数的均值，并根据这个均值生成哈希值。
def get_pHash1(img_path):
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    # plt.imshow(img, cmap='gray')
    # plt.show()

    img = cv2.resize(img, (32, 32), cv2.INTER_CUBIC)
    # plt.imshow(img, cmap='gray')
    # plt.show()

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # plt.imshow(img_gray, cmap='gray')
    # plt.show()

    img_dct = cv2.dct(np.float32(img_gray))

    # 显示DCT系数的图像
    # dct_scaled = cv2.normalize(img_dct, None, 0, 255, cv2.NORM_MINMAX)
    # img_dct_scaled = dct_scaled.astype(np.uint8)
    # plt.imshow(img_dct_scaled, cmap='gray')
    # plt.show()
    
    img_avg = np.mean(img_dct)
    # print(f"DCT变换后图像块的均值={img_avg}")

    img_hash_str = get_img_hash_binary(img_dct, img_avg)
    # print(f"图像的二进制哈希值={img_hash_str}")

    img_hash = get_img_hash(img_hash_str)
    return img_hash


# DCT变换后：将DCT系数的大小显式地调整为8x8，使用8x8的DCT系数块的频率分布，计算调整后的DCT系数的均值，并生成哈希值。
def get_pHash2(img_path):
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    # plt.imshow(img, cmap='gray')
    # plt.show()

    img = cv2.resize(img, (32, 32), cv2.INTER_CUBIC)
    # plt.imshow(img, cmap='gray')
    # plt.show()

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # plt.imshow(img_gray, cmap='gray')
    # plt.show()

    img_dct = cv2.dct(np.float32(img_gray))
    img_dct.resize(8, 8)

    # 显示DCT系数的图像
    # dct_scaled = cv2.normalize(img_dct, None, 0, 255, cv2.NORM_MINMAX)
    # img_dct_scaled = dct_scaled.astype(np.uint8)
    # plt.imshow(img_dct_scaled, cmap='gray')
    # plt.show()

    img_avg = np.mean(img_dct)
    # print(f"DCT变换后图像块的均值={img_avg}")

    img_hash_str = get_img_hash_binary(img_dct, img_avg)
    # print(f"图像的二进制哈希值={img_hash_str}")

    img_hash = get_img_hash(img_hash_str)
    return img_hash


# DCT变换后：只提取DCT系数的左上角8x8块的信息，然后计算这个块的均值。此法只考虑图像一小部分的频率分布，并生成哈希值。
def get_pHash3(img_path):
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    # plt.imshow(img, cmap='gray')
    # plt.show()

    img = cv2.resize(img, (32, 32), cv2.INTER_CUBIC)
    # plt.imshow(img, cmap='gray')
    # plt.show()

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # plt.imshow(img_gray, cmap='gray')
    # plt.show()

    img_dct = cv2.dct(np.float32(img_gray))
    dct_roi = img_dct[0:8, 0:8]

    # 显示DCT系数的图像
    # dct_scaled = cv2.normalize(dct_roi, None, 0, 255, cv2.NORM_MINMAX)
    # img_dct_scaled = dct_scaled.astype(np.uint8)
    # plt.imshow(img_dct_scaled, cmap='gray')
    # plt.show()

    img_avg = np.mean(dct_roi)
    # print(f"DCT变换后图像块的均值={img_avg}")

    img_hash_str = get_img_hash_binary(dct_roi, img_avg)
    # print(f"图像的二进制哈希值={img_hash_str}")

    img_hash = get_img_hash(img_hash_str)
    return img_hash

def get_img_hash_binary(img_dct, img_avg):
    img_hash_str = ''
    for i in range(8):
        img_hash_str += ''.join(map(lambda i: '0' if i < img_avg else '1', img_dct[i]))
    # print(f"图像的二进制哈希值={img_hash_str}")
    return img_hash_str

def get_img_hash(img_hash_str):
    img_hash = ''.join(map(lambda x:'%x' % int(img_hash_str[x : x + 4], 2), range(0, 64, 4)))
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



# ======================================== 测试场景一 ========================================

# img = 'img_test/apple-01.jpg'

# img_hash1 = get_phash1(img)
# img_hash2 = get_phash2(img)
# img_hash3 = get_phash3(img)

# print(f"方式一：DCT变换后，无DCT特征频率区域缩放，获得图像的二进制哈希值={img_hash1}")
# print(f"方式二：DCT变换后，将DCT系数显式调整为8x8，获得图像的二进制哈希值={img_hash2}")
# print(f"方式三：DCT变换后，只提取DCT系数左上角8x8像素，获得图像的二进制哈希值={img_hash3}")



# ======================================== 测试场景二 ========================================

time_start = time.time()

img_1 = 'img_test/apple-01.jpg'
img_2 = 'img_test/apple-02.jpg'
img_3 = 'img_test/apple-03.jpg'
img_4 = 'img_test/apple-04.jpg'
img_5 = 'img_test/apple-05.jpg'
img_6 = 'img_test/apple-06.jpg'
img_7 = 'img_test/apple-07.jpg'
img_8 = 'img_test/apple-08.jpg'
img_9 = 'img_test/apple-09.jpg'
img_10 = 'img_test/pear-001.jpg'

# ------------------------------------- 测试场景二：方式一 --------------------------------------

# img_hash1 = get_pHash1(img_1)
# img_hash2 = get_pHash1(img_2)
# img_hash3 = get_pHash1(img_3)
# img_hash4 = get_pHash1(img_4)
# img_hash5 = get_pHash1(img_5)
# img_hash6 = get_pHash1(img_6)
# img_hash7 = get_pHash1(img_7)
# img_hash8 = get_pHash1(img_8)
# img_hash9 = get_pHash1(img_9)
# img_hash10 = get_pHash1(img_10)

# ------------------------------------- 测试场景二：方式二 --------------------------------------

img_hash1 = get_pHash2(img_1)
img_hash2 = get_pHash2(img_2)
img_hash3 = get_pHash2(img_3)
img_hash4 = get_pHash2(img_4)
img_hash5 = get_pHash2(img_5)
img_hash6 = get_pHash2(img_6)
img_hash7 = get_pHash2(img_7)
img_hash8 = get_pHash2(img_8)
img_hash9 = get_pHash2(img_9)
img_hash10 = get_pHash2(img_10)

# ------------------------------------- 测试场景二：方式三 --------------------------------------

# img_hash1 = get_pHash3(img_1)
# img_hash2 = get_pHash3(img_2)
# img_hash3 = get_pHash3(img_3)
# img_hash4 = get_pHash3(img_4)
# img_hash5 = get_pHash3(img_5)
# img_hash6 = get_pHash3(img_6)
# img_hash7 = get_pHash3(img_7)
# img_hash8 = get_pHash3(img_8)
# img_hash9 = get_pHash3(img_9)
# img_hash10 = get_pHash3(img_10)

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