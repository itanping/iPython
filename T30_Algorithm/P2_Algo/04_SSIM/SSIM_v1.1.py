"""
以图搜图：结构相似性（Structural Similarity，简称SSIM算法）查找相似图像的原理与实现
实验环境：Win10 | python 3.9.13 | OpenCV 4.4.0 | numpy 1.21.1 | Matplotlib 3.7.1
实验时间：2024-01-23
实例名称：SSIM_v1.1.py
"""

"""
1. 图像预处理：读取原始图像与匹配图像，并进行图像灰度处理。若两图有宽高差异，则调整图像维度
2. 计算结构相似度：计算两个灰度图像之间的结构相似度
3. 图像阈值处理：对差异图像进行阈值处理，得到一个二值化图像
4. 查找图像轮廓：在经过阈值处理后的图像中查找轮廓，并将找到的轮廓绘制在一个新的图像上
5. 绘制图像轮廓：检测到的轮廓周围放置矩形，并将处理后的图像进行展示
"""

import os
import cv2
import time
import numpy as np
from skimage.metrics import structural_similarity

# time_start = time.time()

# # 目标图像素材库文件夹路径
# database_dir = '../../P0_Doc/img_data/'

# # 读取查询图像和数据库中的图像
# # img1_path = database_dir + 'iphone15-001.jpg'
# # img2_path = database_dir + 'iphone15-002.jpg'
# img1_path = database_dir + 'car-101.jpg'
# img2_path = database_dir + 'car-102.jpg'



# # 读取图像
# img1 = cv2.imread(img1_path)
# img2 = cv2.imread(img2_path)

# # 将图像转换为灰度图像
# img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
# img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# # 检查图像形状，保证两个图像必须具有相同的尺寸，即相同的高度和宽度
# if img1_gray.shape != img2_gray.shape:
#     # 调整图像大小，使它们具有相同的形状
#     img2_gray = cv2.resize(img2_gray, (img1_gray.shape[1], img1_gray.shape[0]))

# # 计算两个图像之间的结构相似性指数（Structural Similarity Index，简称SSIM）的函数
# (score, diff_img) = structural_similarity(img1_gray, img2_gray, full=True)
# # (score, diff_img) = structural_similarity(img1_gray, img2_gray, win_size=101, full=True)
# # 打印结构相似性指数和差异图像的信息
# print(f"两个灰度图像之间的相似性指数：{score}")
# print(f"两个灰度图像之间的图像结构差异：\n{diff_img}")

def structural_similar(img_resouce_gray, img_target): 
    # 将图像转换为灰度图像
    img_target_gray = cv2.cvtColor(cv2.imread(img_target), cv2.COLOR_BGR2GRAY)

    # 检查图像形状，保证两个图像必须具有相同的尺寸，即相同的高度和宽度
    if img_target_gray.shape != img_resouce_gray.shape:
        # 调整图像大小，使它们具有相同的形状
        img_target_gray = cv2.resize(img_target_gray, (img_resouce_gray.shape[1], img_resouce_gray.shape[0]))

    # 计算两个图像之间的结构相似性指数（Structural Similarity Index，简称SSIM）的函数
    (score, diff_img) = structural_similarity(img_resouce_gray, img_target_gray, full=True)
    # 打印结构相似性指数和差异图像的信息
    # print(f"两个灰度图像之间的相似性指数：{score}")
    # print(f"两个灰度图像之间的图像结构差异：\n{diff_img}")
    return (score, diff_img)

def ssim_image_search(img_resouce, database_paths):
    # 将原图像转换为灰度图像
    img_resouce_gray = cv2.cvtColor(cv2.imread(img_resouce), cv2.COLOR_BGR2GRAY)
    
    # 遍历数据库图像并比较相似度
    similaritys = []
    for img_path in database_paths:
        # 获取相似图像文件路径
        img_target_path = os.path.join(script_dir, database_dir_path, img_path)
        # 提取数据库图像的特征
        ssim = structural_similar(img_resouce_gray, img_target_path)
        # 将结果保存到列表中（仅保留相似值大于等于 0.8 的图像）
        if (ssim[0] >= 0.7):
            similaritys.append((os.path.relpath(img_target_path), ssim))
            # print(f"图像名称：{img_target_path}，与目标图像 {os.path.basename(img_resouce)} 的近似值：{ssim[0]}")
    
    # # 按相似度降序排序
    # similaritys.sort(key=lambda item: item[0], reverse=True)
    return similaritys


if __name__ == "__main__":
    time_start = time.time()

    # 获取当前执行脚本所在目录
    script_dir = os.path.dirname(__file__)
    print(f"script_dir:{script_dir}")

    # 目标图像素材库文件夹路径
    database_dir_path = '../../P0_Doc/img_data/'
    # 指定测试图像文件扩展名
    img_suffix = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']

    # 获取测试图像库中所有文件
    all_files = os.listdir(os.path.join(script_dir, database_dir_path))
    # 筛选测试图像库中指定后缀的图像文件
    img_files = [img_file for img_file in all_files if any(img_file.endswith(suffix) for suffix in img_suffix)]
    
    # 获取测试图像库中所有图像的路径
    img_files_path = [os.path.join(database_dir_path, filename) for filename in img_files]

    # print(f"img_files_path:{img_files_path}")

    # # 获取目标测试图像的全路径
    # img_resouce = os.path.join(script_dir, database_dir_path, 'apple-101.jpg')
    # query_image_path = database_folder_path + 'car-101.jpg'
    
    # 获取目标测试图像的全路径
    img_org_path = os.path.join(script_dir, database_dir_path, 'car-101.jpg')
    # print(f"img_org_path:{img_org_path}")

    # 进行相似图像搜索
    img_search_results = ssim_image_search(img_org_path, img_files_path)

    # 按相似度降序排序
    img_search_results.sort(key=lambda item: item[1][0], reverse=True)
    for similarity in img_search_results:
        print(f"图像名称：{similarity[0]}，与目标图像 {os.path.basename(img_org_path)} 的近似值：{similarity[1][0]}")

    time_end = time.time()
    print(f"耗时：{time_end - time_start}")