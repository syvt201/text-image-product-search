from src.models.grounding_dino import GroundingDinoDetector
from src.models.sam_segment import SamSegmentator
import numpy as np
from PIL import Image

class GroundedSamSegmentator:
    def __init__(self, 
                gd_model_id: str = "IDEA-Research/grounding-dino-tiny", 
                sam_model_id: str = "facebook/sam-vit-base",
                device=None
    ):
        self.device = device
        self.detector = GroundingDinoDetector(gd_model_id, device)
        self.segmentator = SamSegmentator(sam_model_id, device)
        
    def grounded_segmentation(self, 
        image : str | Image.Image,
        labels: str | list[str],
        threshold: float = 0.3,
        polygon_refinement: bool = False,
    ):
        """
        Perform grounded segmentation on the image based on the text prompt(s).
        
        Args:
            image (str or PIL Image): Path to image file or PIL Image
            labels (str or list of str): Text prompt(s) for object detection and segmentation.
            threshold (float): Threshold to keep object detection predictions based on confidence score.
            polygon_refinement (bool): Whether to apply polygon refinement to the segmentation masks.
            
        Returns:
            np.ndarray: The original image as a NumPy array.
            list[dict]: List of segmentation results with masks and bounding boxes.
        """
        if image is None or labels is None:
            raise ValueError("Both `image` and `labels` must be provided")
        
        if isinstance(image, str):
            with Image.open(image) as im:
                image = im.convert("RGB").copy()
        elif isinstance(image, Image.Image):
            image = image if image.mode == "RGB" else image.convert("RGB")
        else:
            raise ValueError("`image` must be a file path or a PIL Image")
                
        detections = self.detector.detect(text=labels, image=image, threshold=threshold)
        results = self.segmentator.segment(image=image, detection_results=detections, polygon_refinement=polygon_refinement)

        return np.array(image), results
        