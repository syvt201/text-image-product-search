# 🚀 Usage Guide

This document explains how to set up and use the project.

---

## 📦 1. Installation

Clone the repo:
```
git clone https://github.com/syvt201/text-image-product-search.git
cd text-image-product-search
```

Install dependencies:
```
pip install -r requirements.txt
```
Create a .env file (see config.py or .env.example for variables):
```
MONGO_URI=mongodb://localhost:27017
MONGO_DB=your_db
FAISS_INDEX_PATH=data/faiss.index
IMG_DIR=data/images
...
```

## 🗄️ 2. Build Database
Add images and build FAISS + MongoDB index:
```
python -m src.pipeline.build_index
```
This will:

- Encode images using **CLIP**.

- Store embeddings in **FAISS (IndexFlatL2)**.

- Save metadata in **MongoDB**.

## 🔎 3. Search

Run a search demo:
```
python -m experiments.search_demo --text "a red car"
```

Options:

- Text search only.

- Image search only.

- Text + image combined search.

## ✂️ 4. Object Detection & Segmentation
You can detect and segment objects from an image using **GroundingDINO + SAM:**
```
python -m experiments.segment_demo --image_path images/jet.jpg --prompt "jet fighter; car"
```
The segmented object can then be re-used as a new query for search.

## 🖥️ 5. Webapp (Gradio UI)
```
python -m webapp.demo
```
Features:
- Search by text, image, or both.

- Upload an image + prompt → segment object → search related images.
