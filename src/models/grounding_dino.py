from transformers import GroundingDinoProcessor, GroundingDinoForObjectDetection, infer_device
from PIL import Image
import string
import torch

class GroundingDinoDetector:
    def __init__(self, model_id: str="IDEA-Research/grounding-dino-tiny", device=None):
        self.device = device or infer_device() 
        self.processor = GroundingDinoProcessor.from_pretrained(model_id)
        self.model = GroundingDinoForObjectDetection.from_pretrained(model_id).to(self.device)
    
    def detect(self, text:str | list[str], image: str | list[str] | Image.Image | list[Image.Image], threshold: float = 0.4, text_threshold: float = 0.3):
        """
        Detect objects in the image based on the text prompt.
        Args:
            text (str or list of str): Text prompt(s) for object detection.
            image (str or list of str or Image or list of Image): Path(s) to image file(s) or PIL Image(s)
            threshold (float): Threshold to keep object detection predictions based on confidence score.
            text_threshold (float): Score threshold to keep text detection predictions.
        Returns:
            list[Dict]: A list of dictionaries, each dictionary containing the
                scores: tensor of confidence scores for detected objects
                boxes: tensor of bounding boxes in [x0, y0, x1, y1] format
                labels: list of text labels for each detected object (will be replaced with integer ids in v4.51.0)
                text_labels: list of text labels for detected objects
        """
        
        if text is None or image is None:
            raise ValueError("Both `text` and `image` must be provided")
        
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
        
        # remove punctuation from text
        if isinstance(text, list):
            text_labels = [label.translate(str.maketrans('', '', string.punctuation)) for label in text]
        else:
            text_labels = [text.translate(str.maketrans('', '', string.punctuation))]
        
        # Preproces
        inputs = self.processor(images=pil_images, text=text_labels, return_tensors="pt").to(self.model.device)
        
        with torch.inference_mode():
            outputs = self.model(**inputs)
        
        # Post-process results
        results = self.processor.post_process_grounded_object_detection(
            outputs=outputs,
            input_ids=inputs.input_ids,
            threshold=threshold,
            text_threshold=text_threshold,
            target_sizes=[pil_images[0].size[::-1]]
        )
        
        return results