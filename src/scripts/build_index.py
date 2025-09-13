import os
from dotenv import load_dotenv
import src.db.faiss_utils as faiss_utils
import src.db.mongodb_utils as mongodb_utils
from src.models.clip_encoder import CLIPEncoder
from datetime import datetime, timezone
from PIL import Image
import numpy as np

load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI")
MONGO_DB = os.getenv("MONGO_DB")
MONGO_COLLECTION_COUNTERS = os.getenv("MONGO_COLLECTION_COUNTERS")
MONGO_COLLECTION_METADATA = os.getenv("MONGO_COLLECTION_METADATA")
MONGO_COLLECTION_FAISS_MAPPING = os.getenv("MONGO_COLLECTION_FAISS_MAPPING")
IMAGE_DIR = os.getenv("IMAGE_DIR")
FAISS_INDEX_SAVE_PATH = os.getenv("FAISS_INDEX_SAVE_PATH")

for var in [MONGODB_URI, MONGO_DB, MONGO_COLLECTION_COUNTERS, MONGO_COLLECTION_METADATA, MONGO_COLLECTION_FAISS_MAPPING, IMAGE_DIR, FAISS_INDEX_SAVE_PATH]:
    if var is None:
        raise ValueError("One or more required environment variables are not set.")
    
def get_image_metadata(file_path):
    stat_info = os.stat(file_path)
    size = stat_info.st_size   # kích thước file (bytes)
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
    
def build_index():
    _, metada_collection = mongodb_utils.connect_to_mongodb(MONGODB_URI, MONGO_DB, MONGO_COLLECTION_METADATA)
    _, counter_collection = mongodb_utils.connect_to_mongodb(MONGODB_URI, MONGO_DB, MONGO_COLLECTION_COUNTERS)
    _, faiss_mapping_collection = mongodb_utils.connect_to_mongodb(MONGODB_URI, MONGO_DB, MONGO_COLLECTION_FAISS_MAPPING)
    
    faiss_index = faiss_utils.create_faiss_index(index_type="FlatL2", embedding_dim=512)
    clip_encoder = CLIPEncoder()
    
    for file in os.listdir(IMAGE_DIR):
        if not file.lower().endswith((".jpg", ".png", ".jpeg")):
            continue
        
        image_path = os.path.join(IMAGE_DIR, file)
        
        
        # add metadata
        inserted_id = mongodb_utils.insert_one(collection=metada_collection, 
                                 document=get_image_metadata(image_path))
        
        # update counter
        counter = mongodb_utils.find_one_and_update(collection=counter_collection,
                                                    query={"_id": "vector_id"},
                                                    update={"$inc": {"seq": 1}},
                                                    upsert=True)
        
        next_id = counter["seq"]
        
        # mapping faiss_id <---> mongo_id
        mongodb_utils.insert_one(collection=faiss_mapping_collection,
                                 document={
                                     "faiss_id": next_id,
                                     "mongo_id": inserted_id 
                                 })
        
        print(f"Inserted {file} -> mongo_id={inserted_id}, faiss_id={next_id}")
        

        # embedding
        _, img_embedding = clip_encoder.encode(text=None, image_path=image_path)
        
        # add to faiss
        faiss_utils.add_embedding_with_id_to_index(index=faiss_index, id_ = np.array([next_id]), embedding=img_embedding)
        
    # save faiss index
    faiss_utils.save_index(faiss_index, FAISS_INDEX_SAVE_PATH)
    
    print(f"Saved faiss index into {FAISS_INDEX_SAVE_PATH}")


if __name__ == "__main__":
    build_index()