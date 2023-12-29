"""
以图搜图：差值哈希算法（Difference Hash Algorithm，简称dHash）的原理与实现
测试环境：win10 | python 3.9.13 | OpenCV 4.4.0 | numpy 1.21.1
实验场景：通过 opencv，使用差值哈希算法查找目标图像素材库中所有相似图像
实验时间：2023-10-31
实验名称：dhash_v5_all.py
"""

# ---------------------------------------------------------------------------------------------------------------------
# 多图测试
# ---------------------------------------------------------------------------------------------------------------------

import os
import cv2
import time

def get_dHash(img_path):
    # 读取图像：通过OpenCV的imread加载RGB图像
    img_rgb = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    # 缩小图像：使用OpenCV的resize函数将图像缩放为9x8像素，采用Cubic插值方法进行图像重采样
    img_resize = cv2.resize(img_rgb, (9, 8), cv2.INTER_CUBIC)
    # 图像灰度化：将彩色图像转换为灰度图像
    img_gray = cv2.cvtColor(img_resize, cv2.COLOR_BGR2GRAY)

    # 计算差异值：获得图像二进制字符串
    img_hash_str = ''
    # 遍历图像的像素，比较相邻像素之间的灰度值，根据强弱增减差异情况生成一个二进制哈希值
    # 外层循环，遍历图像的行（垂直方向），范围是从0到7
    for i in range(8):
        # 内层循环，遍历图像的列（水平方向），范围也是从0到7
        for j in range(8):
            # 比较当前像素 img[i, j] 与下一个像素 img[i, j + 1] 的灰度值
            if img_gray[i, j] > img_gray[i, j + 1]:
                # 如果当前像素的灰度值大于下一个像素的灰度值（灰度值增加），将1添加到名为 hash 的列表中
                img_hash_str += '1'
            else:
                # 否则灰度值弱减，将0添加到名为 hash 的列表中
                img_hash_str += '0'
    # print(f"图像的二进制哈希值={img_hash_str}")

    # 生成哈希值：生成图像可识别哈希值
    img_hash = ''.join(map(lambda x:'%x' % int(img_hash_str[x : x + 4], 2), range(0, 64, 4)))
    return img_hash


# 汉明距离：通过两个等长字符串在相同位置上不同字符的数量，计算两个等长字符串之间的汉明距离
def hamming_distance(str1, str2):
    # 检查这两个字符串的长度是否相同。如果长度不同，它会引发 ValueError 异常，因为汉明距离只适用于等长的字符串
    if len(str1) != len(str2):
        raise ValueError("Input strings must have the same length")
    
    distance = 0
    for i in range(len(str1)):
        # 遍历两个字符串的每个字符，比较它们在相同位置上的值。如果发现不同的字符，将 distance 的值增加 1
        if str1[i] != str2[i]:
            distance += 1
    return distance


# ------------------------------------------------- 测试 -------------------------------------------------
if __name__ == "__main__":
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
    org_img_hash = get_dHash(img_org_path)
    print(f"目标图像：{os.path.relpath(img_org_path)}，图像HASH：{org_img_hash}")

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
        img_hash = get_dHash(img_path)
        # 获取相似图像与目标图像的汉明距离
        distance = hamming_distance(org_img_hash, img_hash)
        # 存储相似图像的相对路径、哈希值、汉明距离
        img_hash_all.append((os.path.relpath(img_path), img_hash, distance))

    for img in img_hash_all:
        print(f"图像名称：{os.path.basename(img[0])}，图像HASH：{img[1]}，与目标图像的近似值（汉明距离）：{img[2]}")

    time_end = time.time()
    print(f"耗时：{time_end - time_start}")