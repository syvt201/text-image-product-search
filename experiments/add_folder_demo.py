import os
from dotenv import load_dotenv
from src.pipeline.build_index import build_index
import argparse

load_dotenv()

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--image_dir", type=str, required=True, help="Path to the folder containing images to be added.")
    parsed_args = args.parse_args()
    image_dir = parsed_args.image_dir
    build_index(image_dir=image_dir, load_faiss_index=True)