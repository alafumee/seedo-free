import torch

# Hugging Face Hub
from huggingface_hub import hf_hub_download

# Grounding DINO
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict
from groundingdino.util.inference import (
    annotate,
    load_image,
    predict,
    load_image_from_array,
)
from track_objects import my_annotate
from PIL import Image

def load_model_hf(repo_id, filename, ckpt_config_filename, device="cpu"):
    cache_config_file = hf_hub_download(repo_id=repo_id, filename=ckpt_config_filename)

    args = SLConfig.fromfile(cache_config_file)
    model = build_model(args)
    args.device = device

    cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
    checkpoint = torch.load(cache_file, map_location="cpu")
    log = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print("Model loaded from {} \n => {}".format(cache_file, log))
    _ = model.eval()
    return model

def find_keypoint_coords(image, obj_caption, save_path = None):
    BOX_TRESHOLD = 0.3
    TEXT_TRESHOLD = 0.25
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    ckpt_repo_id = "ShilongLiu/GroundingDINO"
    ckpt_filenmae = "groundingdino_swinb_cogcoor.pth"
    ckpt_config_filename = "GroundingDINO_SwinB.cfg.py"

    groundingdino_model = load_model_hf(
        ckpt_repo_id, ckpt_filenmae, ckpt_config_filename
    )
    
    best_boxes = []
    best_phrases = []
    best_logits = []
    
    image_src, image = load_image_from_array(image)
    
    boxes, logits, phrases = predict(
            model=groundingdino_model,
            image=image,
            caption=obj_caption,
            box_threshold=BOX_TRESHOLD,
            text_threshold=TEXT_TRESHOLD,
            device=DEVICE,
        )
    if len(boxes) == 0:
        print(">>> No boxes found for the caption:", obj_caption)
        return
    best_boxes.append(boxes[0].unsqueeze(0))
    best_phrases.append(phrases[0])
    best_logits.append(logits[0])
    
    print(">>> GroundingDINO model inference done, found", len(best_boxes), "boxes.")
    print(">>> best_boxes:", best_boxes)
    print(">>> best_phrases:", best_phrases)
    print(">>> best_logits:", best_logits)
    
    # visualize box on image and save
    if len(best_boxes) > 0:
        image = my_annotate(
            image_src,
            torch.concat(best_boxes, dim=0),
            best_logits,
            best_phrases,
        )
        # save the annotated image
        image = Image.fromarray(image)
        image.save(save_path)
    else:
        print(">>> No boxes found.")
        
    return best_boxes[0].cpu().numpy() if len(best_boxes) > 0 else None
    