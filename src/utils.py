import os
from PIL import Image
from datetime import datetime, timezone
import cv2
import numpy as np
import torch 

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
    - polygon (list): List of (x, y) coordinates representing the vertices of the polygon.
    - image_shape (tuple): Shape of the image (height, width) for the mask.

    Returns:
    - np.ndarray: Segmentation mask with the polygon filled.
    """
    # Create an empty mask
    mask = np.zeros(image_shape, dtype=np.uint8)

    # Convert polygon to an array of points
    pts = np.array(polygon, dtype=np.int32)

    # Fill the polygon with white color (255)
    cv2.fillPoly(mask, [pts], color=(255,))

    return mask

def get_boxes(results: list[dict]) -> list[list[list[float]]]:
    # `results` are the detection results from GroundingDino. This is a list of dictionaries, with each dictionary containing the following keys:
    #   "scores: The confidence scores for each predicted box on the image.
    #   "labels: Indexes of the classes predicted by the model on the image.
    #   "boxes: Image bounding boxes in (top_left_x, top_left_y, bottom_right_x, bottom_right_y) format.
    
    list_boxes = []
    for result in results:
        boxes = result["boxes"].detach().cpu().tolist()
        list_boxes.append(boxes)
        
    return list_boxes

def refine_masks(masks : torch.Tensor, polygon_refinement: bool = False):
    masks = masks.detach().cpu().float()
    if masks.ndim == 4:   # [B, C, H, W]
        masks = masks.permute(0, 2, 3, 1).mean(dim=-1)  # [B, H, W]
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