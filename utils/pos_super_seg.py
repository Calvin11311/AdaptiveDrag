import numpy as np
from skimage import io
from skimage.segmentation import slic
from skimage.measure import regionprops
from scipy.spatial import distance

# 加载图像，这里使用 skimage 中的示例数据作为图像源
from skimage.data import coffee

# image = coffee()

# # 使用SLIC进行超像素分割
# segments = slic(image, n_segments=100, compactness=10)

# # 获取每个超像素的属性，包括中心点位置
# regions = regionprops(segments + 1)  # 区域标签从1开始

# # 存储每个超像素中心点的坐标
# centroids = np.array([props.centroid for props in regions])

# # 测试点的坐标 (x, y)
# x, y = 250, 300  # 这里依然使用示例坐标，需要替换为你感兴趣的点的坐标

# # 找到测试点属于的超像素标签，然后找到对应的中心点
# segment_label = segments[x, y]
# point_centroid = centroids[segment_label]

# # 计算该中心点与所有其他超像素中心点的距离
# all_distances = distance.cdist([point_centroid], centroids, 'euclidean').flatten()

# # 移除距离自身的距离（第一个元素），然后找到最近邻的距离
# all_distances = np.delete(all_distances, segment_label)
# nearest_distance = np.min(all_distances)

# print(f"The nearest distance from the center of the segment at position ({x},{y}) to another superpixel center is: {nearest_distance}")

def cal_super_seg_dis(segments,centroids,x,y):
    segment_label = segments[x, y]
    point_centroid = centroids[segment_label]

    # 计算该中心点与所有其他超像素中心点的距离
    all_distances = distance.cdist([point_centroid], centroids, 'euclidean').flatten()

    # 移除距离自身的距离（第一个元素），然后找到最近邻的距离
    all_distances = np.delete(all_distances, segment_label)
    nearest_distance = np.min(all_distances)
    return int(nearest_distance)

    # print(f"The nearest distance from the center of the segment at position ({x},{y}) to another superpixel center is: {nearest_distance}")
