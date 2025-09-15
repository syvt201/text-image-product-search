from pymongo import MongoClient

def insert_one(collection, document):
    """
    Insert a single document into a collection
    
    Args:
        collection: MongoDB collection object
        document (dict): Document to insert
        
    Returns:
        str: Inserted document ID or None if insertion fails
    """
    
    try:
        result = collection.insert_one(document)

        return str(result.inserted_id)
    
    except Exception as e:
        print(f"Error inserting document: {e}")
        return None

def insert_many(collection, documents):
    """
    Insert multiple documents into a collection
    Args:
        collection: MongoDB collection object
        documents (list[dict]): List of documents to insert
        
    Returns:
        list: List of inserted document IDs (strs) or None if insertion fails
    """
    
    try:
        if not documents:
            raise ValueError("The documents list is empty.")
        result = collection.insert_many(documents)
        
        return [str(_id) for _id in result.inserted_ids]
    
    except Exception as e:
        print(f"Error inserting documents: {e}")
        return None

def find_one(collection, query):
    """
    Find a single document in a collection based on a query
    Args:
        collection: MongoDB collection object
        query (dict): Query to find the document
    Returns:
        dict: Found document or None if not found
    """
    
    try:
        document = collection.find_one(query)
        return document
    
    except Exception as e:
        print(f"Error finding document: {e}")
        return None

def find_many(collection, query, limit=-1):
    """
    Find multiple documents in a collection based on a query
    
    Args:
        collection: MongoDB collection object
        query (dict): Query to find the documents
        limit (int): Maximum number of documents to return (0 for no limit)
    
    Returns:
        list: List of found documents
    """
    
    try:
        cursor = collection.find(query, limit=limit)
        return list(cursor)
    
    except Exception as e:
        print(f"Error finding documents: {e}")
        return []
    
def update_one(collection, query, update):
    """
    Update a single document in a collection
    
    Args:
        collection: MongoDB collection object
        query (dict): Query to match the document
        update (dict): Update operations to apply
        
    Returns:
        bool: True if update was successful, False otherwise
    """
    
    try:
        result = collection.update_one(query, update)
        return result.modified_count > 0

    except Exception as e:
        print(f"Error update document: {e}")
        return False
    
def update_many(collection, query, update):
    """
    Update multiple documents in a collection
    
    Args:
        collection: MongoDB collection object
        query (dict): Query to match the documents
        update (dict): Update operations to apply
        
    Returns:
        int: Number of documents updated
    """
    
    try:
        result = collection.update_many(query, update)
        return result.modified_count
    
    except Exception as e:
        print(f"Error update documents: {e}")
        return 0
    
def delete_one(collection, query):
    """
    Delete a single documents from a collection
    
    Args:
        collection: MongoDB collection object
        query (dict): Query to match the documents
        
    Returns:
        bool: True if deletion was successful, False otherwise
    """
    
    try:
        result = collection.delete_one(query)
        return result.deleted_count > 0
    
    except Exception as e:
        print(f"Error deleting document: {e}")
        return False
    
def delete_many(collection, query):
    """
    Delete multiple documents from a collection
    
    Args:
        collection: MongoDB collection object
        query (dict): Query to match the documents
        
    Returns:
        int: Number of deleted documents
    """
    
    try:
        result = collection.delete_many(query)
        return result.deleted_count
    
    except Exception as e:
        print(f"Error deleting documents: {e}")
        return 0
    
def drop_collection(collection):
    """
    Drop a collection from database
    
    Args:
        collection: MongoDB collection object
    
    Returns:
        bool: True if collection was dropped, False otherwise
    """
    
    try:
        collection.drop()
        return True
    
    except Exception as e:
        print(f"Error dropping collection: {e}")
        return False
    
def count_documents(collection, query):
    """
    Count the number of documents in a collection
    
    Args:
        collection: MongoDB collection object
        query (dict): Query to filter documents. Can be empty to count all documents
        
    Returns:
        int: Number of documents matching the query
    """
    
    try:
        return collection.count_documents(query)
    except Exception as e:
        print(f"Error counting documents: {e}")
        return 0
    
def connect_to_mongodb(uri, db_name, collection_name):
    """
    Connect to a MongoDB database and return the database and collection objects
    
    Args:
        uri (str): The MongoDB connection
        db_name (str): The name of database to connect to
        collection_name (str): The name of connect to access within the database
        
    Return:
        tuple: A tuple containing the MongoDB database and collection objects
    """
    
    try:
        client = MongoClient(uri)
        db = client[db_name]
        collection = db[collection_name]
        
        return db, collection
    
    except Exception as e:
        print(f"Error connecting to MongoDB database: {e}")
        return None, None
        
    