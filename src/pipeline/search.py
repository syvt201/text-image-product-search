from src.db import faiss_utils, mongodb_utils
from src.models.clip_encoder import CLIPEncoder
import numpy as np 
from PIL import Image
# text ---> embedding ---> search FAISS ---> get mongo_id ---> get metadata from MongoDB
class SearchPipeline:
    """ Pipeline to search images based on text query. """
    
    def __init__(self, faiss_index, metadata_collection, mapping_collection, clip_encoder):
        self.faiss_index = faiss_index
        self.metadata_collection = metadata_collection
        self.mapping_collection = mapping_collection
        self.clip_encoder = clip_encoder
        
    def search(self, query: str = None, image: str | Image.Image = None, top_k: int = 5, alpha: float = 0.5):
        """
        Search images based on a text query or image.
        Args:
            query (str): The text query to search for. Defaults to None.
            image (str or PIL Image): The path to the image file or a PIL Image to search for. Defaults to None.
            top_k (int): The number of top results to return.
        Returns:
            list: List of metadata documents for the top_k most similar images.
        """
        query = query if query and query.strip() != "" else None
        
        if query is None and image is None:
            raise ValueError("Either query or image must be provided.")
        
        
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer.")
        
        if query is not None:
            if not isinstance(query, str):
                raise ValueError("Query must be a non-empty string.")
            # embed text
            try:
                text_embedding, _ = self.clip_encoder.encode(text=query)
            except Exception as e:
                raise RuntimeError(f"Failed to encode text '{query}': {e}")
        
            # distances, indices = faiss_utils.search_index(index=self.faiss_index, query_embedding=text_embedding, top_k=top_k)
            
        if image is not None:
            # embed image
            if not (isinstance(image, (str, Image.Image))):
                raise ValueError("image_path must be a file path or a PIL Image.")
            try:
                _, image_embedding = self.clip_encoder.encode(image=image)
            except Exception as e:
                raise RuntimeError(f"Failed to encode image: {e}")
            
            # distances, indices = faiss_utils.search_index(index=self.faiss_index, query_embedding=image_embedding, top_k=top_k)
        
        if query is not None and image is not None:
            query_embedding = alpha * image_embedding + (1 - alpha) * text_embedding
        elif query is not None:
            query_embedding = text_embedding
        else:
            query_embedding = image_embedding
        
        distances, indices = faiss_utils.search_index(index=self.faiss_index, query_embedding=query_embedding, top_k=top_k)
        
        indices= indices.flatten().tolist()  # shape (1, top_k) ---> (top_k,)
        distances = distances.flatten().tolist()  
        
        if len(indices) == 0:
            print("No similar images found.")
            return []
        
        mapping_docs = mongodb_utils.find_many(collection=self.mapping_collection, 
                                            query={"faiss_id": {"$in": indices}}, 
                                            limit=0)
        
        inserted_ids = [doc["mongo_id"] for doc in mapping_docs]
        
        metadata_docs = mongodb_utils.find_many(collection=self.metadata_collection,
                                                   query={"_id": {"$in" : inserted_ids}},
                                                   limit=0)
        
        return list(metadata_docs), distances
    