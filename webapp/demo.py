import gradio as gr
from src.models.grounded_sam import GroundedSamSegmentator
from experiments.search_demo import search
import numpy as np
import src.db.faiss_utils as faiss_utils
import src.db.mongodb_utils as mongodb_utils
from src.models.clip_encoder import CLIPEncoder
import src.config as config
from src.pipeline.search import SearchPipeline
import cv2

_, metadata_collection = mongodb_utils.connect_to_mongodb(config.MONGODB_URI, config.MONGO_DB, config.MONGO_COLLECTION_METADATA)
_, faiss_mapping_collection = mongodb_utils.connect_to_mongodb(config.MONGODB_URI, config.MONGO_DB, config.MONGO_COLLECTION_FAISS_MAPPING)

try:
    print("Loading existing faiss index...")
    faiss_index = faiss_utils.load_index(config.FAISS_INDEX_SAVE_PATH)
except Exception as e: # create new one
    raise ValueError("Faiss index not found. Please run the image addition script first to create and populate the index.") 
    
clip_encoder = CLIPEncoder(model_name=config.CLIP_MODEL_NAME)

search_pipeline = SearchPipeline(faiss_index=faiss_index,
                                    metadata_collection=metadata_collection,
                                    mapping_collection=faiss_mapping_collection,
                                    clip_encoder=clip_encoder)
segmentator = GroundedSamSegmentator()

css = """
    #gallery div {
        max-height: 75vh !important;
    }
    
    .hide {
        display: none !important
    }
    
    #result_col {
        max-height: 90vh !important;
    }
    
    #input_image {
        max-height: 400px !important;
    }
    
    #submit_btn {
        background: #4dd2ed !important;
    }
    
    #submit_btn:hover {
        opacity: 0.9;
        cursor: pointer; 
    }
"""

bg_colors = {
    "White": [255, 255, 255],
    "Black": [0, 0, 0],
    "Blue": [0, 128, 255],
    "Pink": [255, 204, 255],
    "Yellow": [255, 236, 141],
}

def smooth_edges(img, ksize=5, sigma=1.0, thresh=0.5):
    blurred = cv2.GaussianBlur(img.astype(np.float32), (ksize, ksize), sigma)
    blurred = blurred / blurred.max()
    smoothed = (blurred > thresh).astype(np.uint8) * 255
    return smoothed

def smooth_contour(mask, epsilon=2.0):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    smooth_mask = np.zeros_like(mask)
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        cv2.fillPoly(smooth_mask, [approx], 255)
    return smooth_mask


def perform_segmentation(image, lbls, selected_bg_color):
    if not lbls:
        return None, None
    label_list = [lbl.strip() for lbl in lbls.split(',')]
    img_array, seg_results = segmentator.grounded_segmentation(image=image, labels=label_list, polygon_refinement=True)

    mask = np.zeros(shape=np.array(image).shape[:-1], dtype=np.uint8)
    for i in range(len(seg_results)):
        mask[seg_results[i].mask > 0] = i + 1
    
    
    mask_img = img_array.copy()
    bg_color = bg_colors[selected_bg_color]
    r, g, b = bg_color
    mask_img[:, :, 0][mask == 0] = r
    mask_img[:, :, 1][mask == 0] = g
    mask_img[:, :, 2][mask == 0] = b
    
    # return mask_img, img_array
    return [mask_img]

def perform_search(query, image, top_k, alpha):
    # documents, distances = search(text=query, image=image if image else None, top_k=k, alpha=alpha)
    documents, distances = search_pipeline.search(query=query, image=image if image else None, top_k=top_k, alpha=alpha)
    return [doc.get('url', '') for doc in documents]

def submit(image, query, k, alp, bg_clr, task):
    if task == "Image Search":
        return {
            output_image: perform_search(query, image, k, alp),
            top_k: gr.update(elem_classes=[]),
            alpha: gr.update(elem_classes=[]),
            bg_selector: gr.update(elem_classes=["hide"])
        }
    else:
        return {
            output_image: perform_segmentation(image, query, bg_clr),
            top_k: gr.update(elem_classes=["hide"]),
            alpha: gr.update(elem_classes=["hide"]),
            bg_selector: gr.update(elem_classes=[])
        }

def toggle_task_selector(task):
    if task == "Image Search":
        return {
            top_k: gr.update(elem_classes=[]),
            alpha: gr.update(elem_classes=[]),
            bg_selector: gr.update(elem_classes=["hide"])
        }
    else:
        return {
            top_k: gr.update(elem_classes=["hide"]),
            alpha: gr.update(elem_classes=["hide"]),
            bg_selector: gr.update(elem_classes=[])
        }

with gr.Blocks(css=css) as demo:
    gr.Markdown("# Image Search and Segmentation")
    with gr.Row(equal_height=True):
        with gr.Column(scale=2):
            text_query = gr.Textbox(label="Text query", placeholder="Enter text to search/segment images...")
            input_image = gr.Image(label="Image", type="pil", elem_id="input_image")
            top_k = gr.Number(label="Top K Results", value=3, precision=0, elem_id="top_k")
            alpha = gr.Slider(label="Image weight", minimum=0.0, maximum=1.0, value=0.3, elem_id="img_search_weight")
            bg_selector = gr.Radio(choices=bg_colors.keys(), label="Background Color", value="White", elem_id="bg_selector", elem_classes=["hide"])
            task_selector = gr.Radio(choices=["Image Search", "Image Segmentation"], label="Task", value="Image Search")
            submit_button = gr.Button("Submit", elem_id="submit_btn")
        with gr.Column(scale=3, elem_id="result_col"):
            output_image = gr.Gallery(label="Result", elem_id="gallery", columns=3, preview=True)
        
    submit_button.click(fn=submit, 
                        inputs=[input_image, text_query, top_k, alpha, bg_selector, task_selector], 
                        outputs=[output_image, top_k, alpha, bg_selector])
    task_selector.select(
        fn=toggle_task_selector,
        inputs=[task_selector],
        outputs=[top_k, alpha, bg_selector]
    )
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=8080, allowed_paths=[config.IMAGE_DIR])