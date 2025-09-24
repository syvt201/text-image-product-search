from transformers import SamModel, SamProcessor, infer_device
from PIL import Image
import torch
import numpy as np
from src.utils.helpers import get_boxes, polygon_to_mask, refine_masks

class SamSegmentator:
    def __init__(self, model_id="facebook/sam-vit-base", device=None):
        self.device = device or infer_device()
        self.processor = SamProcessor.from_pretrained(model_id)
        self.model = SamModel.from_pretrained(model_id).to(self.device)
    
    def segment(self, 
                image: str | Image.Image, 
                detection_results: list[dict] = None,
                polygon_refinement: bool = False
    ):
        """ 
        Segment objects in the image based on detection results.
        
        Args:
            image (str | PIL.Image): The input image or path to the image file.
            detection_results (list[dict]): List of detection results containing bounding boxes.
            polygon_refinement (bool): Whether to refine masks using polygon approximation.   
        
        Returns:
            list[dict]: List of detection results with added segmentation masks.
        """
        
        if image is None or detection_results is None:
            raise ValueError("Both `image` and `detection_results` must be provided")
        
        
        if isinstance(image, str):
            with Image.open(image) as im:
                pil_image = im.convert("RGB").copy()
        elif isinstance(image, Image.Image):
            pil_image = image if image.mode == "RGB" else image.convert("RGB")
        else:
            raise ValueError("`image` must be a file path or a PIL Image")
        if detection_results:
            boxes = get_boxes(results=detection_results)
            inputs = self.processor(images=pil_image, input_boxes=boxes, return_tensors="pt").to(self.device)
        else:
            inputs = self.processor(images=pil_image, return_tensors="pt").to(self.device)
            
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