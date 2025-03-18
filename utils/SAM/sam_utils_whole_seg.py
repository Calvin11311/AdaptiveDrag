from .segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
def show_anns(anns,handle_points):
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
                if(m[h_y][h_x]==True):
                    points_mask[m]=True
                    flag_h[j]=False
    return img,points_mask


def generate_sam_seg(img_path,handle_points):
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
    re_img,handle_seg=show_anns(masks,handle_points)
    cv2.imwrite(save_path, re_img * 255)
    # re_img=cv2.imread(save_path)
    return cv2.imread(save_path),handle_seg