"""
以图搜图：感知哈希算法（Perceptual hash algorithm）的原理与实现
测试环境：win10 | python 3.9.13 | OpenCV 4.4.0 | numpy 1.21.1
实验时间：2023-10-22
"""

# ---------------------------------------------------------------------------------------------------------------------
# 测试图像每一步的变换过程
# ---------------------------------------------------------------------------------------------------------------------

import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 测试图片路径
img_path = 'img_test/apple-01.jpg'
# img_path = 'img_car/X3-08.jpg'

 
# 读取原图：通过OpenCV加载图像
img = cv2.imread(img_path)
plt.imshow(img, cmap='gray')
plt.show()

# 通道重排，从BGR转换为RGB
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img, cmap='gray')
plt.show()

# 缩小图像：使用OpenCV的resize函数将图像缩放为32x32像素，采用Cubic插值方法进行图像重采样
img_32 = cv2.resize(img, (32, 32), cv2.INTER_CUBIC)
plt.imshow(img_32, cmap='gray')
plt.show()

# 图像灰度化：将彩色图像转换为灰度图像。
img_gray = cv2.cvtColor(img_32, cv2.COLOR_BGR2GRAY)
print(f"缩放32x32的图像中每个像素的颜色=\n{img_gray}")
plt.imshow(img_gray, cmap='gray')
plt.show()

# 离散余弦变换（DCT）：计算图像的DCT变换，得到32×32的DCT变换系数矩阵
img_dct = cv2.dct(np.float32(img_gray))
print(f"灰度图像离散余弦变换（DCT）={img_dct}")

# 缩放DCT系数
dct_scaled = cv2.normalize(img_dct, None, 0, 255, cv2.NORM_MINMAX)
img_dct_scaled = dct_scaled.astype(np.uint8)

# 显示DCT系数的图像
plt.imshow(img_dct_scaled, cmap='gray')
plt.show()

# 计算灰度均值：计算DCT变换后图像块的均值
img_avg = np.mean(img_dct)
print(f"DCT变换后图像块的均值={img_avg}")

# 生成二进制哈希值
img_hash_str = ''
for i in range(8):
    for j in range(8):
        if img_dct[i, j] > img_avg:
            img_hash_str += '1'
        else:
            img_hash_str += '0'

print(f"图像的二进制哈希值={img_hash_str}")

# 生成图像可识别哈希值
img_hash = ''
for i in range(0, 64, 4):
    img_hash += ''.join('%x' % int(img_hash_str[i: i + 4], 2))
print(f"图像可识别的哈希值={img_hash}")
