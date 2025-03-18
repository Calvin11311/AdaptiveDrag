from .segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    return img
    ax.imshow(img)

sam = sam_model_registry["vit_h"](checkpoint="./segment-anything-main/sam_vit_h_4b8939.pth")
mask_generator = SamAutomaticMaskGenerator(sam)
t="./segment-anything-main/test_img/panda.png"
print(f"Processing '{t}'...")
image = cv2.imread(t)
if image is None:
    print(f"Could not load '{t}' as an image, skipping...")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
masks = mask_generator.generate(image)
# for i, mask_data in enumerate(masks):
#     mask = mask_data["segmentation"]
#     filename = f"{i}.png"
save_path="./segment-anything-main/sam_re/test.png"
# cv2.imwrite(os.path.join(save_path, filename), mask * 255)
# cv2.imwrite(save_path, masks * 255)


print(len(masks))
print(masks[0].keys())
# plt.figure(figsize=(20,20))
# plt.imshow(image)
re_img=show_anns(masks)
cv2.imwrite(save_path, re_img * 255)
# plt.axis('off')
# plt.show()