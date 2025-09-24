import os
from dotenv import load_dotenv
import src.db.faiss_utils as faiss_utils
import src.db.mongodb_utils as mongodb_utils
from src.models.clip_encoder import CLIPEncoder
import src.config as config
from datetime import datetime, timezone
from PIL import Image
import numpy as np
from src.pipeline.add_image import AddImagePipeline
from src.utils.helpers import get_image_metadata

load_dotenv()

# Load environment variables
    
def build_index(image_dir, load_faiss_index=False, drop_existing_collections=False):
    """ Build or load FAISS index and add images from a directory.
    Args:
        image_dir (str): Directory containing images to add to the index.
        load_faiss_index (bool): Whether to load an existing FAISS index. If False, create a new one.
    """
    
    if image_dir is None:
        return
    
    if not isinstance(image_dir, str):
        raise ValueError("image_dir must be a string.")
    
    if drop_existing_collections:
        mongodb_utils.drop_all_collections(config.MONGODB_URI, config.MONGO_DB)
        print(f"All collections from {config.MONGO_DB} database are droppped")

    # connect or create new collections
    _, metadata_collection = mongodb_utils.connect_to_mongodb(config.MONGODB_URI, config.MONGO_DB, config.MONGO_COLLECTION_METADATA)
    _, counter_collection = mongodb_utils.connect_to_mongodb(config.MONGODB_URI, config.MONGO_DB, config.MONGO_COLLECTION_COUNTERS)
    _, faiss_mapping_collection = mongodb_utils.connect_to_mongodb(config.MONGODB_URI, config.MONGO_DB, config.MONGO_COLLECTION_FAISS_MAPPING)
    
    if load_faiss_index: 
        print("Loading existing faiss index...")
        faiss_index = faiss_utils.load_index(config.FAISS_INDEX_SAVE_PATH)
    else: # create new one
        print("Creating new faiss index...")
        faiss_index = faiss_utils.create_faiss_index(index_type="FlatL2", embedding_dim=512) 
        
    clip_encoder = CLIPEncoder(model_name=config.CLIP_MODEL_NAME)
    
    add_image_pipeline = AddImagePipeline(faiss_index=faiss_index,
                                          metadata_collection=metadata_collection,
                                          counter_collection=counter_collection,
                                          mapping_collection=faiss_mapping_collection,
                                          clip_encoder=clip_encoder)
    
    add_image_pipeline.add_image(image_dir=image_dir, batch_size=64)
        
    # save faiss index
    faiss_utils.save_index(faiss_index, config.FAISS_INDEX_SAVE_PATH)
    
    print(f"Saved faiss index into {config.FAISS_INDEX_SAVE_PATH}")

if __name__ == "__main__":
    build_index(image_dir=config.IMAGE_DIR, load_faiss_index=False, drop_existing_collections=True)