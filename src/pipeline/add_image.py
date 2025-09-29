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
    
    def _process_and_add_images(self, image_paths: list[str]):
        """ Process a batch of images and add them to the index and database. """
        if image_paths is None or len(image_paths) == 0:
            return
        
        if not (isinstance(image_paths, list) and all(isinstance(p, str) for p in image_paths)):
            raise ValueError("image_paths must be a list of strings.")
        
        for img_path in image_paths:
            basename = os.path.basename(img_path)
            if not basename.lower().endswith((".jpg", ".png", ".jpeg")):
                raise ValueError("Unsupported file format. Only .jpg, .png, .jpeg are allowed.")
            
        # embedding first
        try:
            _, img_embeddings = self.clip_encoder.encode(image=image_paths) # img_embeddings will be in cpu (as_numpy=True as default)
        except Exception as e:  
            raise RuntimeError(f"Failed to encode image(s): {e}")
        
        # add metadata batch
        documents = [get_image_metadata(p) for p in image_paths]
        inserted_ids = mongodb_utils.insert_many(collection=self.metadata_collection, documents=documents)
        
        # Generate IDs in batch
        batch_size = len(image_paths)
        counter = mongodb_utils.find_one_and_update(collection=self.counter_collection,
                                                        query={"_id": "vector_id"},
                                                        update={"$inc": {"seq": batch_size}},
                                                        upsert=True)
        start_id = counter["seq"] - batch_size + 1
        next_ids = list(range(start_id, start_id + batch_size))

        # Insert mapping batch
        mapping_docs = [
            {"faiss_id": next_id, "mongo_id": inserted_id}
            for next_id, inserted_id in zip(next_ids, inserted_ids)
        ]
        mongodb_utils.insert_many(collection=self.mapping_collection, documents=mapping_docs)
        
        # Add to Faiss
        faiss_utils.add_embedding_with_id_to_index(
            index=self.faiss_index,
            id_=np.array(next_ids),
            embedding=img_embeddings
        )

        
    def add_image(self, image_dir: str = None, image: str | list[str] = None, batch_size: int = 8):
        """
        Add image(s) to the FAISS index and MongoDB collections.
        
        Args:
            image_dir (str): Directory containing images to add. If provided, `image` is ignored.
            image (str or list of str): Path(s) to image file(s) to add.
            batch_size (int): Number of images to process in a batch when adding from a directory.
        """
        
        if image_dir is None and image is None:
            raise ValueError("At least one of `image_dir` or `image` must be provided.")
        
        if image_dir is not None and image is not None:
            print("Both `image_dir` and `image` are provided. Ignoring `image` and using `image_dir`.")
        
        if batch_size < 1:
            raise ValueError("`batch_size` must be a positive integer")
        
        image_paths = []
        
        if image_dir is not None:
            if not os.path.isdir(image_dir):
                raise ValueError(f"{image_dir} is not a valid directory.")
            
            for file in os.listdir(image_dir):
                if file.lower().endswith((".jpg", ".png", ".jpeg")):
                    image_paths.append(os.path.join(image_dir, file))
            
            if len(image_paths) == 0:
                print(f"No supported image files found in directory {image_dir}.")
                return
            
            # Process in batches
            for i in range(0, len(image_paths), batch_size):
                batch_paths = image_paths[i: min(i + batch_size, len(image_paths))]
                self._process_and_add_images(batch_paths)
                print(f"Inserted {i + batch_size:06d} images into MongoDB and added embeddings to Faiss index.")
        else:
            if isinstance(image, str):
                image_paths = [image]
            elif isinstance(image, list) and all(isinstance(p, str) for p in image):
                image_paths = image
            self._process_and_add_images(image_paths)
            print(f"Inserted {len(image_paths):06d} images into MongoDB and added embeddings to Faiss index.")
            
            
        