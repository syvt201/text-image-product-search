from transformers import SamModel, SamProcessor, infer_device
from PIL import Image
import torch
import numpy as np
from src.utils import mask_to_polygon, get_boxes, polygon_to_mask, refine_masks

class SamSegmentator:
    def __init__(self, model_id="facebook/sam-vit-base", device=None):
        self.device = device or infer_device()
        self.processor = SamProcessor.from_pretrained(model_id)
        self.model = SamModel.from_pretrained(model_id).to(self.device)
    
    def segment(self, 
                image: str | list[str] | Image.Image | list[Image.Image], 
                detection_results: list[dict],
                polygon_refinement: bool = False
    ):
        
        if image is None or detection_results is None:
            raise ValueError("Both `image` and `detection_results` must be provided")
        
        if not isinstance(image, list):
            image = [image]
        
        pil_images = []
        for img in image:
            if isinstance(img, str):
                with Image.open(img) as im:
                    pil_images.append(im.convert("RGB").copy())
            elif isinstance(img, Image.Image):
                pil_images.append(img if img.mode == "RGB" else img.convert("RGB"))
            else:
                raise ValueError("`image` must be a file path, a PIL Image, or a list of either")
            
        boxes = get_boxes(results=detection_results)
        inputs = self.processor(images=pil_images, input_boxes=boxes, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            outputs = self.model(**inputs)
            
        masks = self.processor.post_process_masks(
            masks=outputs.pred_masks,
            original_sizes=inputs.original_sizes,
            reshaped_input_sizes=inputs.reshaped_input_sizes
        )[0]
        
        masks = refine_masks(masks, polygon_refinement)
        for detection_result, mask in zip(detection_results, masks):
            detection_result.mask = mask

        return detection_results 
        
        
                

    