"""Subject grouping and image description using CLIP."""

import os
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from sklearn.cluster import AgglomerativeClustering

from .embeddings import get_text_embeddings, cosine_similarity
from .index import ImageIndex, ImageEntry


# Predefined categories for zero-shot classification
SCENE_LABELS = [
    "a photo at the beach",
    "a photo at a pool",
    "an indoor photo",
    "an outdoor photo",
    "a photo in nature",
    "a photo in a city",
    "a photo at home",
    "a photo at a park",
    "a photo at a restaurant",
    "a photo at an event"
]

SUBJECT_LABELS = [
    "a photo of a person",
    "a photo of people",
    "a photo of a pet",
    "a photo of an animal",
    "a photo of food",
    "a photo of a landscape",
    "a photo of a building",
    "a photo of an object",
    "a selfie",
    "a group photo"
]

ACTIVITY_LABELS = [
    "people swimming",
    "people walking",
    "people sitting",
    "people eating",
    "people celebrating",
    "people playing",
    "people traveling",
    "people working",
    "people relaxing",
    "people posing"
]


@dataclass
class ImageGroup:
    """A group of similar images with a suggested description."""
    images: List[ImageEntry] = field(default_factory=list)
    suggested_name: str = ""
    description: str = ""
    confidence: float = 0.0


def classify_image(embedding: np.ndarray, labels: List[str]) -> Tuple[str, float]:
    """
    Classify an image embedding against text labels using CLIP.

    Args:
        embedding: Image embedding
        labels: List of text descriptions to match against

    Returns:
        Tuple of (best matching label, confidence score)
    """
    text_embeddings = get_text_embeddings(labels)
    similarities = np.dot(text_embeddings, embedding)
    best_idx = np.argmax(similarities)
    return labels[best_idx], float(similarities[best_idx])


def get_image_description(embedding: np.ndarray) -> str:
    """
    Generate a description for an image using CLIP zero-shot classification.

    Args:
        embedding: Image embedding

    Returns:
        Generated description string
    """
    scene, scene_conf = classify_image(embedding, SCENE_LABELS)
    subject, subject_conf = classify_image(embedding, SUBJECT_LABELS)

    # Extract key terms from the descriptions
    scene_term = scene.replace("a photo ", "").replace("an ", "").replace("at the ", "").replace("at a ", "").replace("in a ", "").replace("in ", "").replace("at ", "")
    subject_term = subject.replace("a photo of ", "").replace("a ", "").replace("an ", "")

    return f"{subject_term} - {scene_term}"


def cluster_images(
    index: ImageIndex,
    n_clusters: Optional[int] = None,
    distance_threshold: float = 0.3
) -> List[List[ImageEntry]]:
    """
    Cluster images by embedding similarity using hierarchical clustering.

    Args:
        index: ImageIndex containing images to cluster
        n_clusters: Fixed number of clusters (if None, uses distance_threshold)
        distance_threshold: Maximum linkage distance for clustering (used if n_clusters is None)

    Returns:
        List of clusters, each containing a list of ImageEntry objects
    """
    entries = index.get_all_entries()
    if len(entries) < 2:
        return [entries] if entries else []

    # Stack embeddings into matrix
    embeddings = np.vstack([e.embedding for e in entries])

    # Convert to distance matrix (1 - cosine_similarity)
    # Since embeddings are normalized, cosine_sim = dot product
    similarity_matrix = np.dot(embeddings, embeddings.T)
    distance_matrix = 1 - similarity_matrix

    # Perform hierarchical clustering
    if n_clusters is not None:
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric='precomputed',
            linkage='average'
        )
    else:
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=distance_threshold,
            metric='precomputed',
            linkage='average'
        )

    labels = clustering.fit_predict(distance_matrix)

    # Group entries by cluster label
    clusters: Dict[int, List[ImageEntry]] = {}
    for entry, label in zip(entries, labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(entry)

    return list(clusters.values())


def group_by_subject(
    index: ImageIndex,
    n_clusters: Optional[int] = None,
    distance_threshold: float = 0.3
) -> List[ImageGroup]:
    """
    Group images by subject using clustering and generate descriptions.

    Args:
        index: ImageIndex to group
        n_clusters: Fixed number of groups (if None, auto-determined)
        distance_threshold: Clustering distance threshold (used if n_clusters is None)

    Returns:
        List of ImageGroup objects with suggested names
    """
    clusters = cluster_images(index, n_clusters, distance_threshold)
    groups = []

    for cluster in clusters:
        if not cluster:
            continue

        # Compute average embedding for the cluster
        avg_embedding = np.mean([e.embedding for e in cluster], axis=0)
        avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)

        # Generate description from average embedding
        description = get_image_description(avg_embedding)

        # Classify to get confidence
        _, scene_conf = classify_image(avg_embedding, SCENE_LABELS)
        _, subject_conf = classify_image(avg_embedding, SUBJECT_LABELS)
        confidence = (scene_conf + subject_conf) / 2

        # Create a simplified name for file renaming
        suggested_name = description.split(" - ")[0].replace(" ", "_").lower()

        group = ImageGroup(
            images=cluster,
            suggested_name=suggested_name,
            description=description,
            confidence=confidence
        )
        groups.append(group)

    # Sort groups by size (largest first)
    groups.sort(key=lambda g: len(g.images), reverse=True)
    return groups


def rename_images_by_group(groups: List[ImageGroup], dry_run: bool = True) -> List[Tuple[str, str]]:
    """
    Generate rename operations for grouping images.

    Args:
        groups: List of ImageGroup objects
        dry_run: If True, only return proposed renames without executing

    Returns:
        List of (old_path, new_path) tuples
    """
    renames = []

    for group in groups:
        prefix = group.suggested_name
        for i, entry in enumerate(group.images):
            old_path = entry.path
            directory = os.path.dirname(old_path)
            old_name = entry.filename
            ext = os.path.splitext(old_name)[1]

            new_name = f"{prefix}_{i+1:03d}{ext}"
            new_path = os.path.join(directory, new_name)

            # Avoid overwriting existing files
            counter = 1
            while os.path.exists(new_path) and new_path != old_path:
                new_name = f"{prefix}_{i+1:03d}_{counter}{ext}"
                new_path = os.path.join(directory, new_name)
                counter += 1

            if old_path != new_path:
                renames.append((old_path, new_path))

                if not dry_run:
                    os.rename(old_path, new_path)

    return renames
