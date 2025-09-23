from src.db import faiss_utils as faiss_utils , mongodb_utils as mongodb_utils
from src.utils.helpers import get_image_metadata
import os
import numpy as np

class AddImagePipeline:
    """ Pipeline to add images to the database and FAISS index. """
    
    def __init__(self, faiss_index, metadata_collection, counter_collection, mapping_collection, clip_encoder):
        self.faiss_index = faiss_index
        self.metadata_collection = metadata_collection
        self.counter_collection = counter_collection
        self.mapping_collection = mapping_collection
        self.clip_encoder = clip_encoder
    
    def add_image(self, image_path: str):
        """
        Add an image to the database and FAISS index.
        Args:
            image_path (str): The path to the image file.
        Returns:
            None
        """
        
        basename = os.path.basename(image_path)
        if not basename.lower().endswith((".jpg", ".png", ".jpeg")):
            raise ValueError("Unsupported file format. Only .jpg, .png, .jpeg are allowed.")
        
        # embedding first
        try:
            _, img_embedding = self.clip_encoder.encode(image_paths=image_path)
        except Exception as e:  
            raise RuntimeError(f"Failed to encode image {image_path}: {e}")
        
        # add metadata
        inserted_id = mongodb_utils.insert_one(collection=self.metadata_collection, 
                                                document=get_image_metadata(image_path))
        
        # update counter
        counter = mongodb_utils.find_one_and_update(collection=self.counter_collection,
                                                    query={"_id": "vector_id"},
                                                    update={"$inc": {"seq": 1}},
                                                    upsert=True)
        
        next_id = counter["seq"]
        
        # mapping faiss_id <---> mongo_id
        mongodb_utils.insert_one(collection=self.mapping_collection,
                                    document={
                                    "faiss_id": next_id,
                                    "mongo_id": inserted_id 
                                    })
        
        print(f"Inserted {os.path.basename(image_path)} -> mongo_id={inserted_id}, faiss_id={next_id}")

        # add to faiss
        faiss_utils.add_embedding_with_id_to_index(index=self.faiss_index, id_ = np.array([next_id]), embedding=img_embedding)
        
        print(f"Added image embedding to Faiss index with id {next_id} and updated MongoDB.")