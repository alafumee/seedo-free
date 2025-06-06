import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from PIL import Image
import numpy as np
np.random.seed(3)
import matplotlib.pyplot as plt
def show_mask(mask, ax, random_color=False, borders = True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        import cv2
        contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # Try to smooth contours
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

def show_masks(image, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True):
    print(" drawing masks ---------\n")
    for i, (mask, score) in enumerate(zip(masks, scores)):
        print(f"drawing mask {i+1} ---------\n")
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca(), borders=borders)
        if point_coords is not None:
            assert input_labels is not None
            show_points(point_coords, input_labels, plt.gca())
        if box_coords is not None:
            # boxes
            show_box(box_coords, plt.gca())
        if len(scores) > 1:
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        # plt.show()
    # save the plot
        plt.savefig(f"/home/yunzhe/seedo-free/visualization/white_toothbrush_mask_{i}.jpg")
sam2_checkpoint = "../SeeDo/segment-anything-2/checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

predictor = SAM2ImagePredictor(build_sam2(model_cfg, sam2_checkpoint))
image = Image.open("/home/yunzhe/seedo-free/visualization/start_image_copy_white toothbrush.jpg")
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    predictor.set_image(image)
    ## use bounding box to segment the object
    box_coords = [0.42320734, 0.33557087, 0.31779918, 0.12498938]
    box_coords = np.array(box_coords) * np.array([image.width, image.height, image.width, image.height])
    box_center = round(box_coords[0]), round(box_coords[1])
    box_width = round(box_coords[2])
    box_height = round(box_coords[3])
    box = np.array([box_center[0] - box_width // 2, box_center[1] - box_height // 2, box_center[0] + box_width // 2, box_center[1] + box_height // 2])
    print(box,end="  box\n")
    masks, scores, _ = predictor.predict(box=box)
    print(masks.shape,end="  infer finished\n")
    # save the mask
    show_masks(image, masks, scores, point_coords=None, box_coords=box, input_labels=None, borders=True)

from scaffold import annotate_image
numpy_image , sampled_coords = annotate_image(image, mask=masks[0])
## convert numpy_image to PIL Image
annotated_image = Image.fromarray(numpy_image)

save_path = "/home/yunzhe/seedo-free/visualization/white_toothbrush_mask_annotated.jpg"
annotated_image.save(save_path)
