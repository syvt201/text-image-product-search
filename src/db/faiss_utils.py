import faiss
import numpy as np

def create_faiss_index(embedding_dim, index_type='FlatL2'):
    """
    Create a FAISS index.

    Args:
        embedding_dim (int): Dimension of the embeddings.
        index_type (str): Type of FAISS index to create. Default is 'Flat'.

    Returns:
        faiss.Index: The created FAISS index.
    """
    
    index_types = {
        "FlatL2": faiss.IndexFlatL2,
        "IVFFlat": faiss.IndexIVFFlat,
    }
    if index_type in index_types:
        index = index_types[index_type](embedding_dim)
        index = faiss.IndexIDMap(index)
    else:
        raise ValueError(f"Unsupported index type: {index_type}")
    
    return index

def add_embedding_with_id_to_index(index, id_, embedding):
    """
    Add an embedding with id to a FAISS index.

    Args:
        index (faiss.Index): The FAISS index.
        id_ (int): The ID to associate with the embedding.
        embedding (np.ndarray): Embedding to add to the index.

    Returns:
        None
    """
    
    if not isinstance(embedding, np.ndarray):
        raise ValueError("Embedding must be a numpy array.")
    
    if embedding.ndim != 2 or embedding.shape[1] != index.d:
        raise ValueError(f"Embedding must have shape (n_samples, {index.d}).")
    
    index.add_with_ids(embedding, id_)
    
def add_embeddings_with_ids_to_index(index, ids, embeddings):
    """
    Add an embeddings with id to a FAISS index.

    Args:
        index (faiss.Index): The FAISS index.
        ids (list[int]): The IDs to associate with the embeddings.
        embeddings (np.ndarray): Embeddingss to add to the index.

    Returns:
        None
    """
    
    if not isinstance(embeddings, np.ndarray):
        raise ValueError("Embeddingss must be a numpy array.")
    
    if len(ids) != embeddings.shape[0]:
        raise ValueError("Length of ids must match number of embeddings.")
    
    if embeddings.ndim != 2 or embeddings.shape[1] != index.d:
        raise ValueError(f"Embeddings must have shape (n_samples, {index.d}).")
    
    index.add_with_ids(embeddings, ids)
    
def search_index(index, query_embedding, top_k=5):
    """
    Search the FAISS index for the nearest neighbors of a query embedding.

    Args:
        index (faiss.Index): The FAISS index.
        query_embedding (np.ndarray): Query embedding to search for.
        top_k (int): Number of nearest neighbors to retrieve. Default is 5. 
    Returns:
        tuple: (distances, indices) of the nearest neighbors.
    """
    
    if not isinstance(query_embedding, np.ndarray):
        raise ValueError("Query embedding must be a numpy array.")
    
    if query_embedding.ndim != 2 or query_embedding.shape[1] != index.d:
        raise ValueError(f"Query embedding must have shape (1, {index.d}).")
    
    distances, indices = index.search(query_embedding, top_k)
    
    return distances, indices

def save_index(index, file_path):
    """
    Save the FAISS index to a file.

    Args:
        index (faiss.Index): The FAISS index.
        file_path (str): Path to save the index file.

    Returns:
        None
    """
    
    faiss.write_index(index, file_path)
    
def load_index(file_path):
    """
    Load a FAISS index from a file.

    Args:
        file_path (str): Path to the index file.        
    Returns:
        faiss.Index: The loaded FAISS index.
    """
    
    index = faiss.read_index(file_path)
    return index