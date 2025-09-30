# 🏗️ Project Architecture

This document explains the internal architecture and data flow of the project.

---

## 📖 Overview

The system enables:
- Encoding images/text with **CLIP**.
- Storing embeddings in **FAISS**.
- Storing metadata in **MongoDB**.
- Searching via text, image, or both.
- Object detection + segmentation via **GroundingDINO + SAM**.
- Use the segmented object as the query to search similar objects.
---

## 🔄 Pipeline
**1. Add Image Flow**

**Goal:** Add new images into the system so they can be searched later.

* **Load image** from `IMAGE_DIR`

* **Generate ID & metadata** (filename, category, tags, timestamp, path)

* **Encode image ➝ CLIP embedding**

* **Insert metadata ➝ MongoDB**

* **Insert embedding ➝ FAISS index**


**2. Search Flow**

**Goal:** Search for relevant images using text or image queries.

* **Input query: text or image or both**

* **Encode query ➝ CLIP embedding**
    - If text ➝ CLIP text encoder
    - If image ➝ CLIP image encoder
    - If both ➝ Encode the image and text separately, then combine their vectors using the formula: a.image_vector + (1-a).text_vector, where 0 ≤ a ≤ 1.
- **If text ➝ CLIP text encoder**

- **If image ➝ CLIP image encoder**

- **Search FAISS index for top-k nearest neighbors**

- **Fetch metadata (image path, tags, info) from MongoDB**

- **Return ranked results (images + metadata)**

**3. Segmentation Flow**

**Goal:** Segment specific objects from image.

* **Select image (from search results or upload)**

* **Provide prompt (text description)**

* **Grounding DINO ➝ detect regions of interest**

* **Segment Anything Model (SAM) ➝ generate mask**

* **Apply mask ➝ output cropped/segmented object**

## 🧩Components

**Models**

* [`clip_encoder.py`](src/models/clip_encoder.py) – Encodes image/text.

* [`grounding_dino.py`](src/models/grounding_dino.py) – Object detection.

* [`sam_segment.py`](src/models/sam_segment.py) – Segmentation.

* [`grounded_sam.py`](src/models/grounded_sam.py) – Combined pipeline.

**Database**

* [`faiss_utils.py`](src/db/faiss_utils.py)

    Provides helper functions to:
    - Initialize and configure FAISS index.
    - Insert embedding vectors.
    - Perform nearest-neighbor search (using **L2 distance - Euclidean**).
    - Save/load index to disk.

* [`mongodb_utils.py`](src/db/mongodb_utils.py)

    Provides helper functions to:
    - Manage MongoDB collections.
    - Insert and retrieve image metadata.
    - Handle auto-increment counters for image IDs.
    - Map FAISS vectors with metadata entries.
  
**Pipelines**

* [`add_image.py`](src/pipeline/add_image.py) – Batch ingestion.

* [`search.py`](src/pipeline/search.py) – Search pipeline.

**Webapp**

* [`demo.py`](webapp/demo.py)– Gradio UI.