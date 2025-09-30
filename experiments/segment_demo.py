from src.models.grounded_sam import GroundedSamSegmentator
from src.utils.helpers import annotate
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFilter
import argparse
import os

argparser = argparse.ArgumentParser()
argparser.add_argument('--image_path', type=str, default="images/jet.jpg", help='Path to the input image')
argparser.add_argument('--prompt', type=str, default="cars", help='List of labels to segment, separated by semicolon (;)')

args = argparser.parse_args()
if not os.path.exists(args.image_path):
    raise FileNotFoundError(f"Image path {args.image_path} does not exist.")

image_path = args.image_path
labels = [lbl for lbl in args.prompt.split(';') if lbl.strip()]

gs = GroundedSamSegmentator()
image, segment_results = gs.grounded_segmentation(image=image_path, labels=labels, threshold=0.3, polygon_refinement=True) 

annotated_image = annotate(image, segment_results)
mask = np.zeros(shape=image.shape[:-1], dtype=np.uint8)
for i in range(len(segment_results)):
    mask[segment_results[i].mask > 0] = i + 1

# smooth
pil_mask = Image.fromarray(mask)
pil_mask = pil_mask.filter(ImageFilter.ModeFilter(size=3))

plt.figure(figsize=(12, 6))
plt.subplot(2, 2, 1)
plt.imshow(image)
plt.title("Image")
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(annotated_image)
plt.title("Segmented image")
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(mask)
plt.title("Mask")
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(np.array(pil_mask))
plt.title("Smoothen Mask")
plt.axis('off')


plt.tight_layout()
plt.show()