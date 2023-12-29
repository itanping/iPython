"""
以图搜图：差值哈希算法（Difference Hash Algorithm，简称dHash）的原理与实现
测试环境：win10 | python 3.9.13 | OpenCV 4.4.0 | numpy 1.21.1
实验时间：2023-10-25
"""

# ---------------------------------------------------------------------------------------------------------------------
# 多图测试
# ---------------------------------------------------------------------------------------------------------------------

import cv2
import time

def get_dHash(img_path):
    # 读取图像：通过OpenCV的imread加载图像
    img = cv2.imread(img_path)

    # 通道重排：从BGR转换为RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 缩小图像：使用OpenCV的resize函数将图像缩放为9x8像素，采用Cubic插值方法进行图像重采样
    img_resize = cv2.resize(img_rgb, (9, 8), cv2.INTER_CUBIC)

    # 图像灰度化：将彩色图像转换为灰度图像
    img_gray = cv2.cvtColor(img_resize, cv2.COLOR_BGR2GRAY)

    # 计算差异值：获得图像二进制字符串
    img_hash_str = ''
    # img_hash_arr = []
    # 遍历图像的像素，比较相邻像素之间的灰度值，根据强弱增减差异情况生成一个二进制哈希值
    # 外层循环，遍历图像的行（垂直方向），范围是从0到7
    for i in range(8):
        # 内层循环，遍历图像的列（水平方向），范围也是从0到7
        for j in range(8):
            # 比较当前像素 img[i, j] 与下一个像素 img[i, j + 1] 的灰度值
            if img_gray[i, j] > img_gray[i, j + 1]:
                # 如果当前像素的灰度值大于下一个像素的灰度值（灰度值增加），将1添加到名为 hash 的列表中
                # img_hash_arr.append(1)
                img_hash_str += '1'
            else:
                # 否则灰度值弱减，将0添加到名为 hash 的列表中
                # img_hash_arr.append(0)
                img_hash_str += '0'
    # print(f"图像的二进制哈希值={img_hash_str}")

    # 生成哈希值：生成图像可识别哈希值
    img_hash = ''
    for i in range(0, 64, 4):
        img_hash += ''.join('%x' % int(img_hash_str[i: i + 4], 2))
    # print(f"图像可识别的哈希值={img_hash}")
    return img_hash


# 汉明距离：通过两个等长字符串在相同位置上不同字符的数量，计算两个等长字符串之间的汉明距离
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

img_hash1 = get_dHash(img_1)
img_hash2 = get_dHash(img_2)
img_hash3 = get_dHash(img_3)
img_hash4 = get_dHash(img_4)
img_hash5 = get_dHash(img_5)
img_hash6 = get_dHash(img_6)
img_hash7 = get_dHash(img_7)
img_hash8 = get_dHash(img_8)
img_hash9 = get_dHash(img_9)
img_hash10 = get_dHash(img_10)

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