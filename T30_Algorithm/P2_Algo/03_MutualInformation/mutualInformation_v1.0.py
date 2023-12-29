import cv2
import numpy as np

# 目标图像素材库文件夹路径
database_dir = '../../P0_Doc/img_data/'

# 读取查询图像和数据库中的图像
img1_path = database_dir + 'car-101.jpg'
img2_path = database_dir + 'car-102.jpg'
img2_path = database_dir + 'car-103.jpg'

# 读取两幅图像
img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

# 使用 OpenCV 中的 calcHist 函数计算直方图
hist1 = cv2.calcHist([img1], [0], None, [256], [0, 256])
hist2 = cv2.calcHist([img2], [0], None, [256], [0, 256])

# 将直方图归一化
hist1 /= hist1.sum()
hist2 /= hist2.sum()

# 计算互信息
mutual_info = np.sum(np.minimum(hist1, hist2))

print(f"Mutual Information: {mutual_info}")

# 在实际应用中，可以根据互信息的阈值来判断两幅图像是否相似
# 设置互信息的阈值
threshold = 0.7
if mutual_info > threshold:
    print("Images are similar.")
else:
    print("Images are not similar.")
