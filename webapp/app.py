import gradio as gr
from src.models.grounded_sam import GroundedSamSegmentator
from scripts.search_demo import search
import numpy as np
import src.db.faiss_utils as faiss_utils
import src.db.mongodb_utils as mongodb_utils
from src.models.clip_encoder import CLIPEncoder
import src.config as config
from src.pipeline.search import SearchPipeline

_, metadata_collection = mongodb_utils.connect_to_mongodb(config.MONGODB_URI, config.MONGO_DB, config.MONGO_COLLECTION_METADATA)
_, faiss_mapping_collection = mongodb_utils.connect_to_mongodb(config.MONGODB_URI, config.MONGO_DB, config.MONGO_COLLECTION_FAISS_MAPPING)

try:
    print("Loading existing faiss index...")
    faiss_index = faiss_utils.load_index(config.FAISS_INDEX_SAVE_PATH)
except Exception as e: # create new one
    raise ValueError("Faiss index not found. Please run the image addition script first to create and populate the index.") 
    
clip_encoder = CLIPEncoder(model_name=config.CLIP_MODEL_NAME)

add_image_pipeline = SearchPipeline(faiss_index=faiss_index,
                                    metadata_collection=metadata_collection,
                                    mapping_collection=faiss_mapping_collection,
                                    clip_encoder=clip_encoder)

css = """
    #search_gallery div {
        max-height: 80vh !important;
    }
"""

bg_colors = {
    "White": [255, 255, 255],
    "Black": [0, 0, 0],
    "Blue": [0, 128, 255],
    "Pink": [255, 204, 255],
    "Yellow": [255, 236, 141],
}

def create_app():
    segmentator = GroundedSamSegmentator()
    with gr.Blocks(css=css) as app:
        gr.Markdown("# Image Search and Segmentation")
        
        with gr.Tab("Search Images"):
            with gr.Row(equal_height=True):
                with gr.Column(scale=1):
                    text_query = gr.Textbox(label="Text Query", placeholder="Enter text to search images...")
                    image_query = gr.Image(label="Image Query", type="pil")
                    top_k = gr.Number(label="Top K Results", value=6, precision=0)
                    alpha = gr.Slider(label="Image weight", minimum=0.0, maximum=1.0, value=0.3)
                    search_button = gr.Button("Search")
                with gr.Column(scale=2):
                    search_results = gr.Gallery(label="Search Results", elem_id="search_gallery", columns=3, preview=True)
                    
            def perform_search(query, image, k, alpha):
                # documents, distances = search(text=query, image=image if image else None, top_k=k, alpha=alpha)
                documents, distances = add_image_pipeline.search(query=query, image=image if image else None, top_k=k, alpha=alpha)
                return [doc.get('url', '') for doc in documents]
            
            search_button.click(fn=perform_search, inputs=[text_query, image_query, top_k, alpha], outputs=search_results)
        
        with gr.Tab("Grounded Segmentation"):
            with gr.Row():
                with gr.Column(scale=2):
                    seg_image = gr.Image(label="Input Image", type="pil")
                    labels = gr.Textbox(label="Labels", placeholder="Enter comma-separated labels for segmentation...")
                    threshold = gr.Slider(label="Confidence Threshold", minimum=0.0, maximum=1.0, value=0.3)
                    # polygon_refinement = gr.Checkbox(label="Polygon Refinement", value=False)
                    bg_selector = gr.Radio(choices=bg_colors.keys(), label="Background Color", value="White")
                    segment_button = gr.Button("Segment")
                    
                with gr.Column(scale=3):
                    seg_output_image = gr.Image(label="Segmented Image", type="numpy")
                    seg_output_mask = gr.Image(label="Segmentation Mask", type="numpy")
                    
            def perform_segmentation(image, lbls, thresh, selected_bg_color):
                if not lbls:
                    return None, None
                label_list = [lbl.strip() for lbl in lbls.split(',')]
                img_array, seg_results = segmentator.grounded_segmentation(image=image, labels=label_list, threshold=thresh, polygon_refinement=True)

                mask = np.zeros(shape=np.array(image).shape[:-1], dtype=np.uint8)
                for i in range(len(seg_results)):
                    mask[seg_results[i].mask > 0] = i + 1
                
                mask_img = img_array.copy()
                
                bg_color = bg_colors[selected_bg_color]
                r, g, b = bg_color
                mask_img[:, :, 0][mask == 0] = r
                mask_img[:, :, 1][mask == 0] = g
                mask_img[:, :, 2][mask == 0] = b
                    
                return img_array, mask_img
            
            segment_button.click(fn=perform_segmentation, inputs=[seg_image, labels, threshold, bg_selector], outputs=[seg_output_image, seg_output_mask])
        
    return app

if __name__ == "__main__":
    app = create_app()
    app.launch(server_name="0.0.0.0", server_port=8080, allowed_paths=["/mnt/e/coco/coco2017/test2017"])