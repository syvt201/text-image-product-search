import os
from PIL import Image
from datetime import datetime, timezone
import cv2
import numpy as np
import torch 
from dataclasses import dataclass

@dataclass
class BoundingBox:
    xmin: int
    ymin: int
    xmax: int
    ymax: int

    @property
    def xyxy(self) -> list[float]:
        return [self.xmin, self.ymin, self.xmax, self.ymax]

@dataclass
class DetectionResult:
    score: float
    label: str
    box: BoundingBox
    mask: np.array = None

    @classmethod
    def from_dict(cls, detection_dict: dict) -> 'DetectionResult':
        return cls(score=detection_dict['score'],
                   label=detection_dict['label'],
                   box=BoundingBox(xmin=detection_dict['box']['xmin'],
                                   ymin=detection_dict['box']['ymin'],
                                   xmax=detection_dict['box']['xmax'],
                                   ymax=detection_dict['box']['ymax']))
        
def get_image_metadata(file_path):
    stat_info = os.stat(file_path)
    size = stat_info.st_size   # (bytes)
    uploaded_at = datetime.fromtimestamp(stat_info.st_mtime, tz=timezone.utc)
    
    with Image.open(file_path) as img:
        width, height = img.size
        format = img.format.lower()  # jpg, png, webp,...

    return {
        "file_name": os.path.basename(file_path),
        "url": os.path.abspath(file_path),
        "uploaded_at": uploaded_at.isoformat(),
        "size": size,
        "format": format,
        "width": width,
        "height": height
    }
    
def mask_to_polygon(mask: np.ndarray) -> list[list[int]]:
    """
    Convert a binary mask to a polygon representation.
    
    Args:
        mask (np.ndarray): Binary mask where the object is represented by 1s and
                          the background by 0s.
                          
    Returns:
        list: List of (x, y) coordinates representing the vertices of the polygon.
    """
    # Find contours in the binary mask
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the contour with the largest area
    largest_contour = max(contours, key=cv2.contourArea)

    # Extract the vertices of the contour
    polygon = largest_contour.reshape(-1, 2).tolist()

    return polygon

def polygon_to_mask(polygon: list[tuple[int, int]], image_shape: tuple[int, int]) -> np.ndarray:
    """
    Convert a polygon to a segmentation mask.

    Args:
        polygon (list): List of (x, y) coordinates representing the vertices of the polygon.
        image_shape (tuple): Shape of the image (height, width) for the mask.

    Returns:
        np.ndarray: Segmentation mask with the polygon filled.
    """
    # Create an empty mask
    mask = np.zeros(image_shape, dtype=np.uint8)

    # Convert polygon to an array of points
    pts = np.array(polygon, dtype=np.int32)

    # Fill the polygon with white color (255)
    cv2.fillPoly(mask, [pts], color=(255,))

    return mask

def get_boxes(results: DetectionResult) -> list[list[list[float]]]:
    boxes = []
    for result in results:
        xyxy = result.box.xyxy
        boxes.append(xyxy)

    return [boxes]

def refine_masks(masks : torch.Tensor, polygon_refinement: bool = False):
    masks = masks.detach().cpu().float()
    if masks.ndim == 4:   # [B, C, H, W]
        masks = masks.permute(0, 2, 3, 1)  # [B, H, W, C]
    else:
        raise ValueError(f"Unexpected mask shape: {masks.shape}")
    
    masks = masks.mean(axis=-1)
    masks = (masks > 0).int()
    masks = masks.numpy().astype(np.uint8)
    masks = list(masks)

    if polygon_refinement:
        for idx, mask in enumerate(masks):
            shape = mask.shape
            polygon = mask_to_polygon(mask)
            mask = polygon_to_mask(polygon, shape)
            masks[idx] = mask

    return masks

def annotate(image: Image.Image | np.ndarray, detection_results: list[DetectionResult]) -> np.ndarray:
    """
    Annotate the image with bounding boxes and masks from detection results.
    
    Args:
        image (PIL.Image or np.ndarray): The input image to annotate.
        detection_results (list of DetectionResult): List of detection results containing bounding boxes and masks
    
    Returns:
        np.ndarray: Annotated image in RGB format.
    """
    # Convert PIL Image to OpenCV format
    image_cv2 = np.array(image) if isinstance(image, Image.Image) else image
    image_cv2 = cv2.cvtColor(image_cv2, cv2.COLOR_RGB2BGR)

    # Iterate over detections and add bounding boxes and masks
    for detection in detection_results:
        label = detection.label
        score = detection.score
        box = detection.box
        mask = detection.mask

        # Sample a random color for each detection
        color = np.random.randint(0, 256, size=3)

        # Draw bounding box
        cv2.rectangle(image_cv2, (box.xmin, box.ymin), (box.xmax, box.ymax), color.tolist(), 2)
        cv2.putText(image_cv2, f'{label}: {score:.2f}', (box.xmin, box.ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color.tolist(), 2)

        # If mask is available, apply it
        if mask is not None:
            # Convert mask to uint8
            mask_uint8 = (mask * 255).astype(np.uint8)
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(image_cv2, contours, -1, color.tolist(), 2)

    return cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)

def smooth_mask(mask: np.ndarray | Image.Image, kernel_size: int = 5) -> np.ndarray:
    """
    Apply Gaussian smoothing to the mask.
    
    Args:
        mask (np.ndarray or PIL Image): The input mask to smooth.
        kernel_size (int): Size of the Gaussian kernel. Must be an odd integer.
        
    Returns:
        np.ndarray: Smoothed image in RGB format.
    """
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be an odd integer.")
    if isinstance(mask, Image.Image):
        mask = np.array(mask)
        
    if len(mask.shape) == 3 and mask.shape[2] == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
    elif len(mask.shape) != 2:
        raise ValueError("mask must be a 2D array or a 3-channel RGB image.")
    
    smoothed_image = cv2.GaussianBlur(mask, (kernel_size, kernel_size), 0)
    return smoothed_image