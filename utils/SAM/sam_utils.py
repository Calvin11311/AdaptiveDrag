from .segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

def cut_mask(mask,h_x,h_y,t_x,t_y):
    h, w = mask.shape
    y, x = np.ogrid[:h, :w]
    angle = np.arctan2(y - h_y, x - h_x)
    angle_pt = np.arctan2(t_y - h_y, t_x - h_x)
    angle_diff = np.abs(np.angle(np.exp(1j * (angle - angle_pt))))
    angle_region = (angle_diff <= np.pi / 2)
    mask_cut = angle_region & mask
    return mask_cut

def show_anns(anns,handle_points,target_points):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    flag_h=[True]*len(handle_points)
    # for i in range(len(handle_points)):
    #     flag_h.append(True)
    points_mask = np.zeros((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1]),dtype=bool)
    for ann in sorted_anns:
        m = ann['segmentation']
        #0.35？
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    for i in range(len(sorted_anns)-1,-1,-1):
        m=sorted_anns[i]['segmentation']
        for j,h_j in enumerate(handle_points):
            if(flag_h[j]==True):
                h_y=int(h_j[0]*2)
                h_x=int(h_j[1]*2)
                t_y=int(target_points[j][0]*2)
                t_x=int(target_points[j][1]*2)
                if(m[h_y][h_x]==True):
                    cur_mask=cut_mask(m,h_x,h_y,t_x,t_y)
                    points_mask[cur_mask]=True
                    flag_h[j]=False
    return img,points_mask


def generate_sam_seg(img_path,handle_points,target_points):
    # sam = sam_model_registry["vit_h"](checkpoint="./segment-anything-main/sam_vit_h_4b8939.pth")
    sam = sam_model_registry["vit_l"](checkpoint="./segment-anything-main/sam_vit_l_0b3195.pth")
    mask_generator = SamAutomaticMaskGenerator(sam)
    print(f"Processing '{img_path}'...")
    image = cv2.imread(img_path)
    if image is None:
        print(f"Could not load '{img_path}' as an image, skipping...")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    masks = mask_generator.generate(image)
    save_path="./sam_seg.png"
    print(len(masks))
    print(masks[0].keys())
    re_img,handle_seg=show_anns(masks,handle_points,target_points)
    cv2.imwrite(save_path, re_img * 255)
    # re_img=cv2.imread(save_path)
    return cv2.imread(save_path),handle_seg

