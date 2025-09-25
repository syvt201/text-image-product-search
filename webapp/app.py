import gradio as gr
from src.models.grounded_sam import GroundedSamSegmentator
from scripts.search_demo import search

def create_app():
    segmentator = GroundedSamSegmentator()
    with gr.Blocks() as app:
        gr.Markdown("# Image Search and Segmentation")
        
        with gr.Tab("Search Images"):
            with gr.Row():
                with gr.Column():
                    text_query = gr.Textbox(label="Text Query", placeholder="Enter text to search images...")
                    image_query = gr.Image(label="Image Query", type="pil")
                    top_k = gr.Number(label="Top K Results", value=5, precision=0)
                    search_button = gr.Button("Search")
                with gr.Column():
                    search_results = gr.Gallery(label="Search Results")
                    
            def perform_search(query, image, k):
                documents, distances = search(text=query, image=image if image else None, top_k=k)
                return [doc.get('url', '') for doc in documents]
            
            search_button.click(fn=perform_search, inputs=[text_query, image_query, top_k], outputs=search_results)
        
        with gr.Tab("Grounded Segmentation"):
            with gr.Row():
                with gr.Column():
                    seg_image = gr.Image(label="Input Image", type="pil")
                    labels = gr.Textbox(label="Labels", placeholder="Enter comma-separated labels for segmentation...")
                    threshold = gr.Slider(label="Confidence Threshold", minimum=0.0, maximum=1.0, value=0.3)
                    polygon_refinement = gr.Checkbox(label="Polygon Refinement", value=False)
                    segment_button = gr.Button("Segment")
                with gr.Column():
                    seg_output_image = gr.Image(label="Segmented Image")
                    seg_output_data = gr.JSON(label="Segmentation Data")
                    
            def perform_segmentation(image, lbls, thresh, poly_refine):
                if not lbls:
                    return None, None
                label_list = [lbl.strip() for lbl in lbls.split(',')]
                img_array, seg_results = segmentator.grounded_segmentation(image=image, labels=label_list, threshold=thresh, polygon_refinement=poly_refine)
                return img_array, seg_results
            
            segment_button.click(fn=perform_segmentation, inputs=[seg_image, labels, threshold, polygon_refinement], outputs=[seg_output_image, seg_output_data])
        
    return app

if __name__ == "__main__":
    app = create_app()
    app.launch(server_name="0.0.0.0", server_port=8080, allowed_paths=["/mnt/e/coco/coco2017/test2017"])