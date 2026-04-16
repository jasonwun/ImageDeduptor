"""CLIP embedding generation for images."""

import torch
import clip
import numpy as np
from PIL import Image
from typing import Optional, List, Tuple

# Global model state (lazy loaded)
_device = None
_model = None
_preprocess = None


def get_device() -> str:
    """Get the device to use for inference."""
    global _device
    if _device is None:
        _device = "cuda" if torch.cuda.is_available() else "cpu"
    return _device


def load_model():
    """Load the CLIP model (lazy loading)."""
    global _model, _preprocess
    if _model is None:
        device = get_device()
        _model, _preprocess = clip.load("ViT-B/32", device=device)
    return _model, _preprocess


def get_image_embedding(image_path: str) -> Optional[np.ndarray]:
    """
    Convert an image to a normalized CLIP embedding.

    Args:
        image_path: Path to the image file

    Returns:
        Normalized embedding as numpy array, or None if failed
    """
    model, preprocess = load_model()
    device = get_device()

    try:
        image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = model.encode_image(image)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.cpu().numpy().flatten()
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None


def get_text_embeddings(texts: List[str]) -> np.ndarray:
    """
    Convert text descriptions to CLIP embeddings.

    Args:
        texts: List of text descriptions

    Returns:
        Array of normalized embeddings, shape (len(texts), embedding_dim)
    """
    model, _ = load_model()
    device = get_device()

    tokens = clip.tokenize(texts).to(device)
    with torch.no_grad():
        emb = model.encode_text(tokens)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy()


def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """
    Compute cosine similarity between two embeddings.

    Args:
        emb1: First embedding (normalized)
        emb2: Second embedding (normalized)

    Returns:
        Cosine similarity score
    """
    return float(np.dot(emb1, emb2))


# --- Perceptual Hashing ---

def compute_dhash(image_path: str, hash_size: int = 16) -> Optional[np.ndarray]:
    """
    Compute difference hash (dHash) of an image.

    dHash compares adjacent pixels to detect edges/gradients,
    making it robust to scaling and minor color changes but
    sensitive to actual visual structure.

    Args:
        image_path: Path to image file
        hash_size: Size of hash (hash_size x hash_size bits)

    Returns:
        Binary hash as numpy array, or None if failed
    """
    try:
        img = Image.open(image_path).convert('L')  # Grayscale
        # Resize to hash_size+1 width to get hash_size horizontal differences
        img = img.resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
        pixels = np.array(img, dtype=np.float64)

        # Compare adjacent pixels (is left > right?)
        diff = pixels[:, 1:] > pixels[:, :-1]
        return diff.flatten()
    except Exception:
        return None


def compute_ahash(image_path: str, hash_size: int = 16) -> Optional[np.ndarray]:
    """
    Compute average hash (aHash) of an image.

    aHash compares each pixel to the average, detecting
    overall brightness patterns.

    Args:
        image_path: Path to image file
        hash_size: Size of hash (hash_size x hash_size bits)

    Returns:
        Binary hash as numpy array, or None if failed
    """
    try:
        img = Image.open(image_path).convert('L')  # Grayscale
        img = img.resize((hash_size, hash_size), Image.Resampling.LANCZOS)
        pixels = np.array(img, dtype=np.float64)

        # Compare to average
        avg = pixels.mean()
        return (pixels > avg).flatten()
    except Exception:
        return None


def hash_similarity(hash1: Optional[np.ndarray], hash2: Optional[np.ndarray]) -> float:
    """
    Compute similarity between two perceptual hashes.

    Uses normalized Hamming distance.

    Args:
        hash1: First hash
        hash2: Second hash

    Returns:
        Similarity score between 0 and 1 (1 = identical)
    """
    if hash1 is None or hash2 is None:
        return 0.0

    # Hamming distance = number of differing bits
    hamming_dist = np.sum(hash1 != hash2)
    max_dist = len(hash1)

    # Convert to similarity (0 distance = 1.0 similarity)
    return 1.0 - (hamming_dist / max_dist)
