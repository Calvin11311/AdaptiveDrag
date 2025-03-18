import sys
sys.path.append("./utils/sam2/")
sys.path.append("./")
from .sam2.build_sam import build_sam2
from .sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2


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
                    # 防止单点点在中心进行整体拖拽效果不佳的情况发生
                    if((np.sum(cur_mask)/np.sum(m))>=0.5):
                        points_mask[m]=True
                    else:
                        points_mask[cur_mask]=True
                    flag_h[j]=False
    return img,points_mask


def generate_sam2_seg(img_path,handle_points,target_points):

    image = Image.open(img_path)
    image = np.array(image.convert("RGB"))

    sam2_checkpoint = "./utils/sam2/checkpoints/sam2_hiera_base_plus.pt"
    model_cfg = "sam2_hiera_b+.yaml"

    sam2 = build_sam2(model_cfg, sam2_checkpoint, device ='cuda', apply_postprocessing=False)

    mask_generator = SAM2AutomaticMaskGenerator(sam2)

    masks = mask_generator.generate(image)

    save_path="./sam2_seg.png"
    print(len(masks))
    if len(masks)==0:
        handle_seg = np.zeros((image.shape[0], image.shape[1]),dtype=bool)
        cv2.imwrite(save_path, handle_seg * 255)
        return cv2.imread(save_path),handle_seg
    print(masks[0].keys())
    re_img,handle_seg=show_anns(masks,handle_points,target_points)
    cv2.imwrite(save_path, re_img * 255)
    # re_img=cv2.imread(save_path)
    return cv2.imread(save_path),handle_seg
