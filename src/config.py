import os
from dotenv import load_dotenv

load_dotenv()

# Load environment variables
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
MONGO_DB = os.getenv("MONGO_DB", "mydb")

MONGO_COLLECTION_COUNTERS = os.getenv("MONGO_COLLECTION_COUNTERS", "counter")
MONGO_COLLECTION_METADATA = os.getenv("MONGO_COLLECTION_METADATA", "metadata")
MONGO_COLLECTION_FAISS_MAPPING = os.getenv("MONGO_COLLECTION_FAISS_MAPPING", "faiss-mapping")

IMAGE_DIR = os.getenv("IMAGE_DIR", "images")
FAISS_INDEX_SAVE_PATH = os.getenv("FAISS_INDEX_SAVE_PATH", "faiss_index.bin")

CLIP_MODEL_NAME = os.getenv("CLIP_MODEL_NAME", "openai/clip-vit-base-patch32")
SAM_MODEL_PATH = os.getenv("SAM_MODEL_PATH", "./models/sam.pth")