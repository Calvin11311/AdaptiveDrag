from skimage.segmentation import slic, mark_boundaries
from skimage.io import imread, imsave
import numpy as np
import cv2
def get_seg_mask(image,n_segments,compactness,start,target):
    # 应用 SLIC 算法进行超像素分割
    # n_segments 是期望的超像素数量
    # compactness 是用来平衡颜色相似性和空间接近性的权重
    segments = slic(image, n_segments=n_segments, compactness=compactness)
    out = mark_boundaries(image, segments,(255,255,255))
    imsave('./seg_image.png', out)
    mask = np.zeros(segments.shape, dtype=bool)
    mask_start = np.zeros(segments.shape, dtype=bool)
    for i in range(len(start)):
        A = start[i]
        B = target[i]
        #保存单独起始点的mask区域
        start_sub_mask=np.zeros(segments.shape, dtype=bool)
        start_segment_id = segments[int(A[0])*2, int(A[1])*2]
        start_sub_mask[segments == start_segment_id] = True

        sub_mask_image_start = np.zeros(image.shape[:2], dtype=np.float16)
        sub_mask_image_start[start_sub_mask] = 255.0
        # imsave('./mask_images_start/mask_image_'+str(i)+'.png', sub_mask_image_start)
        mask_start+=start_sub_mask
    # 初始化一个全为 False 的数组用作 Mask
        sub_mask = np.zeros(segments.shape, dtype=bool)
        # 检查区域内的每个超像素，并更新 Mask
        a=int(A[0])
        b=int(B[0])
        if(a > b): a,b = b,a
        c=int(A[1])
        d=int(B[1])
        if(c>d): c,d = d,c
        n=b*2+1-a*2
        for t in range(n+1):
            y=int(a*2+t*(b*2+1-a*2)/n)
            x=int(c*2+t*(d*2+1-c*2)/n)
            segment_id = segments[y, x]
            sub_mask[segments == segment_id] = True
        # for y in range(a*2, b*2+1,2):
        #     for x in range(c*2, d*2+1,2):
        #         segment_id = segments[y, x]
        #         sub_mask[segments == segment_id] = True
        # sub_mask_image = np.zeros(image.shape[:2], dtype=np.float16)
        # sub_mask_image[sub_mask] = 255
        # imsave('./mask_images/mask_image_'+str(i)+'.png', sub_mask_image)
        mask+=sub_mask

    # 保存 mask 图像，其中 mask 区域为白色，非 mask 区域为黑色
    mask_image = np.zeros(image.shape[:2], dtype=np.float16)
    mask_image[mask] = 255
    # imsave('./mask_arr_image.png', mask_image)
    #存储仅包含起始点位的mask图像
    # start_mask_image = np.zeros(image.shape[:2], dtype=np.float16)
    # start_mask_image[mask_start] = 255
    # imsave('./mask_image_start.png', start_mask_image)
    return mask_image, segments


def get_boundary_mask(image,n_segments,compactness,start):
    segments = slic(image, n_segments=n_segments, compactness=compactness)
    out = mark_boundaries(image, segments,(255,255,255))
    imsave('./seg_image.png', out)
    mask = np.zeros(segments.shape, dtype=bool)
    mask_start = np.zeros(segments.shape, dtype=bool)
    for i in range(len(start)):
        if(i!=len(start)-1):
            A = start[i]
            B = start[i+1]
        else:
            A = start[i]
            B = start[0]
        #保存单独起始点的mask区域
        start_sub_mask=np.zeros(segments.shape, dtype=bool)
        start_segment_id = segments[int(A[0])*2, int(A[1])*2]
        start_sub_mask[segments == start_segment_id] = True
        sub_mask_image_start = np.zeros(image.shape[:2], dtype=np.float16)
        sub_mask_image_start[start_sub_mask] = 255
        # imsave('./mask_images_start/mask_image_'+str(i)+'.png', sub_mask_image_start)
        mask_start+=start_sub_mask
        sub_mask = np.zeros(segments.shape, dtype=bool)
        # 检查区域内的每个超像素，并更新 Mask
        a=int(A[0])
        b=int(B[0])
        if(a > b): a,b = b,a
        c=int(A[1])
        d=int(B[1])
        if(c>d): c,d = d,c
        n=b*2+1-a*2
        for t in range(n+1):
            y=int(a*2+t*(b*2+1-a*2)/n)
            x=int(c*2+t*(d*2+1-c*2)/n)
            segment_id = segments[y, x]
            sub_mask[segments == segment_id] = True
        mask+=sub_mask
    # 保存 mask 图像，其中 mask 区域为白色，非 mask 区域为黑色
    mask_image = np.zeros(image.shape[:2], dtype=np.float16)
    mask_image[mask] = 255
    return mask_image, segments

def move_mask(original_mask, start_point, target_point):
    # Create an empty mask with the same shape as original_mask
    if np.sum(original_mask.numpy())==0:
        return original_mask
    final_mask = np.zeros_like(original_mask)
    for i in range(len(start_point)):
        cur_start=start_point[i].numpy().tolist()
        cur_start=[int(i)*2 for i in cur_start]
        cur_target=target_point[i].numpy().tolist()
        cur_target=[int(i)*2 for i in cur_target]
        height, width = original_mask.shape[:2]
        moved_mask = np.zeros_like(original_mask)

        # Calculate the translation vector
        delta_x = int(cur_target[1]) - int(cur_start[1])
        delta_y = int(cur_target[0]) - int(cur_start[0])
        trajectory_mask = np.zeros_like(original_mask)
        # Move the mask
        for i in range(height):
            for j in range(width):
                if original_mask[i, j] > 0:  # Only consider the mask area
                    new_x = j + delta_x
                    new_y = i + delta_y
                    # Check if new coordinates are within bounds
                    if 0 <= new_x < width and 0 <= new_y < height:
                        moved_mask[new_y, new_x] = original_mask[i, j]
                        cur_trajectory_mask = np.zeros_like(original_mask)
                        cv2.line(cur_trajectory_mask, [j, i], [new_x, new_y], 1, thickness=1)
                        trajectory_mask = np.maximum(trajectory_mask, cur_trajectory_mask)

        # # Mark the original area
        # trajectory_mask = np.zeros_like(original_mask)
        # cv2.line(trajectory_mask, cur_start, cur_target, 1, thickness=1)

        # Combine original, moved and trajectory masks
        cur_final_mask = np.maximum(original_mask, moved_mask)
        cur_final_mask = np.maximum(cur_final_mask, trajectory_mask)
        final_mask = np.maximum(cur_final_mask, final_mask)
        # final_mask+=cur_final_mask

    return final_mask