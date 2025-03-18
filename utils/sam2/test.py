import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

def show_anns(anns, borders=True):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.5]])
        img[m] = color_mask 
        if borders:
            import cv2
            contours, _ = cv2.findContours(m.astype(np.uint8),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
            # Try to smooth contours
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            cv2.drawContours(img, contours, -1, (0,0,1,0.4), thickness=1) 
    return img
image = Image.open('./segment-anything-main/test_img/bear_1.png')
image = np.array(image.convert("RGB"))

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

# sam2_checkpoint = "./segment-anything-2-main/checkpoints/sam2_hiera_large.pt"
# model_cfg = "sam2_hiera_l.yaml"
# sam2_checkpoint = "./segment-anything-2-main/checkpoints/sam2_hiera_base_plus.pt"
# model_cfg = "sam2_hiera_b+.yaml"
# sam2_checkpoint = "./segment-anything-2-main/checkpoints/sam2_hiera_small.pt"
# model_cfg = "sam2_hiera_s.yaml"
sam2_checkpoint = "./segment-anything-2-main/checkpoints/sam2_hiera_tiny.pt"
model_cfg = ""


sam2 = build_sam2(model_cfg, sam2_checkpoint, device ='cuda', apply_postprocessing=False)

mask_generator = SAM2AutomaticMaskGenerator(sam2)


masks = mask_generator.generate(image)
import cv2
save_path="./segment-anything-2-main/test_bear_t.png"
print(len(masks))
print(masks[0].keys())
re_img=show_anns(masks)
cv2.imwrite(save_path, re_img * 255)
