# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Image duplicate detection tool using OpenAI's CLIP model (ViT-B/32) to find visually similar images via embedding cosine similarity. Features a Tkinter desktop GUI with persistent indexing, duplicate comparison, and content-based grouping.

## Running the Tool

**GUI Application:**
```bash
python main.py
```

**Legacy CLI (for reference):**
```bash
python check_images.py
```

## Dependencies

`torch`, `clip`, `Pillow`, `tqdm`, `numpy`, `scikit-learn`

## Architecture

```
duplicationDetector/
├── main.py              # Entry point, launches GUI
├── gui/
│   ├── __init__.py
│   ├── app.py           # Main Tkinter application window
│   └── dialogs.py       # Custom dialogs for results/grouping
├── core/
│   ├── __init__.py
│   ├── embeddings.py    # CLIP embedding generation
│   ├── index.py         # Index creation, persistence, loading
│   ├── comparison.py    # Duplicate detection logic
│   └── grouping.py      # Subject grouping and renaming
└── check_images.py      # Original CLI script (kept for reference)
```

## Core Modules

### embeddings.py
- `get_image_embedding(path)`: Converts image to normalized CLIP embedding
- `get_text_embeddings(texts)`: Converts text descriptions to CLIP embeddings
- `cosine_similarity(emb1, emb2)`: Compute similarity between embeddings

### index.py
- `ImageIndex` class for managing collections of image embeddings
- `create_from_directory(path)`: Scan and embed all images
- `save(filepath)` / `load(filepath)`: Persist/load index as .pkl file
- `add_images(paths)`: Incrementally add new images

### comparison.py
- `find_duplicates_in_index(index, threshold)`: O(n^2) pairwise comparison within index
- `find_duplicates_against_index(new_paths, index, threshold)`: Compare new images against existing index

### grouping.py
- `group_by_subject(index)`: Cluster images by similarity, generate descriptions using CLIP zero-shot classification
- `rename_images_by_group(groups)`: Rename files with group prefix

## GUI Features

- **Create Index**: Scan directory, build index, save to .pkl file
- **Load Index**: Load existing index from file
- **Compare Images**: Select new images, find duplicates against loaded index
- **Find Duplicates in Index**: Find duplicate pairs within the loaded index
- **Group by Subject**: Cluster images and suggest group names for renaming
- **Threshold Setting**: Adjustable similarity threshold (0.90-0.999)
