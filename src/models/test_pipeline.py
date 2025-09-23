from src.models.grounded_sam import GroundedSamSegmentator
from src.utils import annotate
import matplotlib.pyplot as plt

labels = ["a cat.", "a remote control."]
img_path = "/mnt/e/text-image-product-search/images/000000039769.jpg"

gs = GroundedSamSegmentator()
image, detections = gs.grounded_segmentation(image=img_path, labels=labels, threshold=0.3, polygon_refinement=True) 

annotated_image = annotate(image, detections)
plt.imshow(annotated_image)
plt.axis('off')
save_name = "output.png"
if save_name:
    plt.savefig(save_name, bbox_inches='tight')
plt.show()