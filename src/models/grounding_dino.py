from transformers import pipeline, infer_device
from PIL import Image
import string
import torch
from src.utils import DetectionResult

class GroundingDinoDetector:
    def __init__(self, model_id: str = "IDEA-Research/grounding-dino-tiny", device=None):
        self.device = device or infer_device() 
        self.object_detector = pipeline(model=model_id, task="zero-shot-object-detection", device=self.device)
        
    def detect(self, text:str | list[str], image: str | Image.Image, threshold: float = 0.3):
        """
        Detect objects in the image based on the text prompt.
        
        Args:
            text (str or list of str): Text prompt(s) for object detection.
            image (str or or Image ): Path to image file or PIL Image
            threshold (float): Threshold to keep object detection predictions based on confidence score.
        
        Returns:
            list[dict[str, Any]]
        """
        
        if text is None or image is None:
            raise ValueError("Both `text` and `image` must be provided")
        
        if not isinstance(image, (str, Image.Image)):
            raise ValueError("`image` must be a file path or a PIL Image")
            
        if isinstance(image, str):
            with Image.open(image) as im:
                pil_image = im.convert("RGB").copy()
        else:
            pil_image = image if image.mode == "RGB" else image.convert("RGB")
        
        # remove punctuation from text
        if isinstance(text, list):
            text_labels = [label.translate(str.maketrans('', '', string.punctuation)) for label in text]
        elif isinstance(text, str):
            text_labels = [text.translate(str.maketrans('', '', string.punctuation))]
        else:
            raise ValueError("`text` must be a string or a list of strings")
        
        text_labels = [label + "." for label in text_labels]
        
        results = self.object_detector(pil_image, candidate_labels=text_labels, threshold=threshold)
        
        results = [DetectionResult.from_dict(result) for result in results]

        return results