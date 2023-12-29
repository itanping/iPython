"""
以图搜图：差值哈希算法（Difference Hash Algorithm，简称dHash）的原理与实现
测试环境：win10 | python 3.9.13 | OpenCV 4.4.0 | numpy 1.21.1
实验时间：2023-10-25
"""

# ---------------------------------------------------------------------------------------------------------------------
# 单图测试：图像每一步的变换过程（不看图像）
# ---------------------------------------------------------------------------------------------------------------------

import cv2
import numpy as np
import matplotlib.pyplot as plt

# 测试图片路径
img_path = 'img_test/apple-01.jpg'
 
# 读取图像：通过OpenCV的imread加载图像
img = cv2.imread(img_path)

# 通道重排，从BGR转换为RGB
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
print(f"图像的二进制哈希值={img_hash_str}")

# 生成哈希值：生成图像可识别哈希值
img_hash = ''
for i in range(0, 64, 4):
    img_hash += ''.join('%x' % int(img_hash_str[i: i + 4], 2))
print(f"图像可识别的哈希值={img_hash}")