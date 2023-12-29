"""
以图搜图：余弦相似度（Cosine Similarity）查找相似图像的原理与实现
实验环境：Win10 | python 3.9.13 | OpenCV 4.4.0 | numpy 1.21.1 | Matplotlib 3.7.1
实验时间：2023-11-30
实例名称：imgCosineSimilarity_v1.0_show.py
"""

import os
import cv2
import matplotlib.pyplot as plt

# 目标图像素材库文件夹路径
database_dir = '../../P0_Doc/img_data/'

# 读取查询图像和数据库中的图像
img1_path = database_dir + 'car-101.jpg'
img2_path = database_dir + 'car-102.jpg'
img3_path = database_dir + 'car-103.jpg'
img4_path = database_dir + 'car-106.jpg'
img5_path = database_dir + 'car-109.jpg'

# 读取图像
img1 = cv2.imread(img1_path)
img2 = cv2.imread(img2_path)
img3 = cv2.imread(img3_path)
img4 = cv2.imread(img4_path)
img5 = cv2.imread(img5_path)

# 将图像转换为灰度图像
img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
img3_gray = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
img4_gray = cv2.cvtColor(img4, cv2.COLOR_BGR2GRAY)
img5_gray = cv2.cvtColor(img5, cv2.COLOR_BGR2GRAY)

# 绘制子图
plt.figure(figsize=(12, 4))
# 绘制灰度图像
plt.subplot(1, 5, 1)
plt.imshow(img1_gray, cmap='gray')
plt.title(os.path.basename(img1_path))
plt.subplot(1, 5, 2)
plt.imshow(img2_gray, cmap='gray')
plt.title(os.path.basename(img2_path))
plt.subplot(1, 5, 3)
plt.imshow(img3_gray, cmap='gray')
plt.title(os.path.basename(img3_path))
plt.subplot(1, 5, 4)
plt.imshow(img4_gray, cmap='gray')
plt.title(os.path.basename(img4_path))
plt.subplot(1, 5, 5)
plt.imshow(img5_gray, cmap='gray')
plt.title(os.path.basename(img5_path))
plt.tight_layout()
# 显示灰度图像
plt.show()



# 计算图像的直方图
img1_hist = cv2.calcHist([img1_gray], [0], None, [256], [0, 256])
img2_hist = cv2.calcHist([img2_gray], [0], None, [256], [0, 256])
img3_hist = cv2.calcHist([img3_gray], [0], None, [256], [0, 256])
img4_hist = cv2.calcHist([img4_gray], [0], None, [256], [0, 256])
img5_hist = cv2.calcHist([img5_gray], [0], None, [256], [0, 256])

# 获取图像的特征向量
vector1 = img1_hist.flatten()
vector2 = img2_hist.flatten()
vector3 = img3_hist.flatten()
vector4 = img4_hist.flatten()
vector5 = img5_hist.flatten()




# 使用垂直线（stem lines）绘制向量
plt.figure(figsize=(8, 4))

# 绘制向量1
plt.subplot(1, 5, 1)
plt.stem(vector1)
plt.title('Vector 1')

# 绘制向量2
plt.subplot(1, 5, 2)
plt.stem(vector2)
plt.title('Vector 2')

# 绘制向量3
plt.subplot(1, 5, 3)
plt.stem(vector3)
plt.title('Vector 3')

# 绘制向量4
plt.subplot(1, 5, 4)
plt.stem(vector4)
plt.title('Vector 4')

# 绘制向量5
plt.subplot(1, 5, 5)
plt.stem(vector5)
plt.title('Vector 5')

# 图像向量可视化
plt.tight_layout()
plt.show()




# 使用折线图绘制向量
plt.plot(vector1, color='orange', label='Vector 1')
plt.plot(vector2, color='red', label='Vector 2')
plt.plot(vector3, color='purple', label='Vector 3')
plt.plot(vector4, color='blue', label='Vector 4')
plt.plot(vector5, color='green', label='Vector 5')

# 添加图例
plt.legend()
# 图像向量可视化
plt.show()





# 使用散点图绘制向量
plt.figure(figsize=(8, 4))

# 绘制散点图
plt.scatter(range(len(vector1)), vector1, label='Vector 1', marker='o', s=10)
plt.scatter(range(len(vector2)), vector2, label='Vector 2', marker='x', s=10)
plt.scatter(range(len(vector3)), vector3, label='Vector 3', marker='o', s=10)
plt.scatter(range(len(vector4)), vector4, label='Vector 4', marker='o', s=10)
plt.scatter(range(len(vector5)), vector5, label='Vector 5', marker='o', s=10)

plt.title('Scatter Plot of Vectors')
plt.xlabel('Index')
plt.ylabel('Value')
# 添加图例
plt.legend()
# 图像向量可视化
plt.show()