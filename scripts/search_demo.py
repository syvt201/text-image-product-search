import os
from dotenv import load_dotenv
import src.db.faiss_utils as faiss_utils
import src.db.mongodb_utils as mongodb_utils
from src.models.clip_encoder import CLIPEncoder
import src.config as config
from PIL import Image
import numpy as np
from src.pipeline.search import SearchPipeline

load_dotenv()

def search(text = None, image: str | Image.Image = None, top_k = 5):
    """ Search images based on a text query or image.
    Args:
        text (str, Optional): The text query to search for. Defaults to None.
        image (str, optional): The path to the image file or a PIL Image to search for. Defaults to None.
        top_k (int): The number of top results to return.
    Returns:    
        list: List of metadata documents for the top_k most similar images.
    """
    if text is None and image is None:
        return []
    
    if not isinstance(text, str) and text is not None:
        raise ValueError("text must be a string.")
    
    if not isinstance(image, (str, Image.Image)) and image is not None:
        raise ValueError("image_path must be a string or Image.")
    
    _, metadata_collection = mongodb_utils.connect_to_mongodb(config.MONGODB_URI, config.MONGO_DB, config.MONGO_COLLECTION_METADATA)
    # _, counter_collection = mongodb_utils.connect_to_mongodb(config.MONGODB_URI, config.MONGO_DB, config.MONGO_COLLECTION_COUNTERS)
    _, faiss_mapping_collection = mongodb_utils.connect_to_mongodb(config.MONGODB_URI, config.MONGO_DB, config.MONGO_COLLECTION_FAISS_MAPPING)
    
    try:
        print("Loading existing faiss index...")
        faiss_index = faiss_utils.load_index(config.FAISS_INDEX_SAVE_PATH)
    except Exception as e: # create new one
        print("Creating new faiss index...")
        faiss_index = faiss_utils.create_faiss_index(index_type="FlatL2", embedding_dim=512) 
        
    clip_encoder = CLIPEncoder(model_name=config.CLIP_MODEL_NAME)
    
    add_image_pipeline = SearchPipeline(faiss_index=faiss_index,
                                        metadata_collection=metadata_collection,
                                        mapping_collection=faiss_mapping_collection,
                                        clip_encoder=clip_encoder)
    
    results, distances = add_image_pipeline.search(query=text, image=image, top_k=top_k)
    return results, distances

if __name__ == "__main__":
    # Example usage
    # text_query = "The girl wearing a red spaghetti strap top"
    text_query = "car and motor"
    # image_path = "scripts/hat.jpg"
    image_path = None
    top_k = 5
    
    results, distances = search(text=text_query, image=os.path.abspath(image_path) if image_path else None, top_k=top_k)
    
    for i, (doc, dist) in enumerate(zip(results, distances)):
        print(f"Result {i+1}:")
        print(f"Distance: {dist}")
        print(f"Metadata: {doc}")
        print("-" * 20)