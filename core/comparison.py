"""Duplicate detection logic using embedding comparison."""

import os
from typing import List, Tuple, Optional, Callable
from dataclasses import dataclass, field

from .embeddings import get_image_embedding, cosine_similarity, compute_dhash, hash_similarity
from .index import ImageIndex


@dataclass
class DuplicateMatch:
    """Represents a duplicate match between two images."""
    image1_path: str
    image2_path: str
    similarity: float  # CLIP similarity
    phash_similarity: float = 0.0  # Perceptual hash similarity

    @property
    def image1_name(self) -> str:
        return os.path.basename(self.image1_path)

    @property
    def image2_name(self) -> str:
        return os.path.basename(self.image2_path)

    @property
    def combined_score(self) -> float:
        """Combined score giving more weight to perceptual hash."""
        # pHash is more reliable for actual duplicates, weight it higher
        return 0.3 * self.similarity + 0.7 * self.phash_similarity


def find_duplicates_in_index(
    index: ImageIndex,
    threshold: float = 0.995,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    use_hybrid: bool = True,
    clip_prefilter: float = 0.85,
    phash_threshold: float = 0.90
) -> List[DuplicateMatch]:
    """
    Find duplicate images within an index.

    Performs O(n^2) pairwise comparison of all images in the index.
    When use_hybrid=True, uses CLIP as a pre-filter then verifies with perceptual hash.

    Args:
        index: ImageIndex to search for duplicates
        threshold: Minimum similarity score for final results (0.0-1.0)
        progress_callback: Optional callback(current, total) for progress updates
        use_hybrid: If True, use CLIP + perceptual hash hybrid approach
        clip_prefilter: CLIP threshold for pre-filtering candidates (only if use_hybrid)
        phash_threshold: Perceptual hash threshold for final filtering (only if use_hybrid)

    Returns:
        List of DuplicateMatch objects for pairs above threshold
    """
    entries = index.get_all_entries()
    n = len(entries)
    total_comparisons = (n * (n - 1)) // 2
    duplicates = []
    comparison_count = 0

    # Cache for perceptual hashes (computed on demand)
    phash_cache = {}

    def get_phash(path: str):
        if path not in phash_cache:
            phash_cache[path] = compute_dhash(path)
        return phash_cache[path]

    for i in range(n):
        for j in range(i + 1, n):
            comparison_count += 1

            if progress_callback:
                progress_callback(comparison_count, total_comparisons)

            # Skip excluded pairs (using relative paths)
            rel_path1 = index.get_relative_path(entries[i].path)
            rel_path2 = index.get_relative_path(entries[j].path)
            if index.is_excluded(rel_path1, rel_path2, exclusion_type="duplicate"):
                continue

            clip_sim = cosine_similarity(entries[i].embedding, entries[j].embedding)

            if use_hybrid:
                # Pre-filter with CLIP (lower threshold to catch candidates)
                if clip_sim < clip_prefilter:
                    continue

                # Compute perceptual hash similarity for candidates
                hash1 = get_phash(entries[i].path)
                hash2 = get_phash(entries[j].path)
                phash_sim = hash_similarity(hash1, hash2)

                # Only include if perceptual hash passes threshold
                if phash_sim >= phash_threshold:
                    duplicates.append(DuplicateMatch(
                        image1_path=entries[i].path,
                        image2_path=entries[j].path,
                        similarity=clip_sim,
                        phash_similarity=phash_sim
                    ))
            else:
                # Original CLIP-only behavior
                if clip_sim >= threshold:
                    duplicates.append(DuplicateMatch(
                        image1_path=entries[i].path,
                        image2_path=entries[j].path,
                        similarity=clip_sim,
                        phash_similarity=0.0
                    ))

    # Sort by combined score (or CLIP similarity if not hybrid)
    if use_hybrid:
        duplicates.sort(key=lambda x: x.phash_similarity, reverse=True)
    else:
        duplicates.sort(key=lambda x: x.similarity, reverse=True)
    return duplicates


def find_duplicates_against_index(
    new_image_paths: List[str],
    index: ImageIndex,
    threshold: float = 0.995,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    use_hybrid: bool = True,
    clip_prefilter: float = 0.85,
    phash_threshold: float = 0.90
) -> List[DuplicateMatch]:
    """
    Compare new images against an existing index to find duplicates.

    When use_hybrid=True, uses CLIP as a pre-filter then verifies with perceptual hash.

    Args:
        new_image_paths: Paths to new images to compare
        index: Existing ImageIndex to compare against
        threshold: Minimum similarity score to consider a duplicate (0.0-1.0)
        progress_callback: Optional callback(current, total, filename) for progress updates
        use_hybrid: If True, use CLIP + perceptual hash hybrid approach
        clip_prefilter: CLIP threshold for pre-filtering candidates (only if use_hybrid)
        phash_threshold: Perceptual hash threshold for final filtering (only if use_hybrid)

    Returns:
        List of DuplicateMatch objects where new images match indexed images
    """
    duplicates = []
    index_entries = index.get_all_entries()
    total = len(new_image_paths)

    # Cache for perceptual hashes
    phash_cache = {}

    def get_phash(path: str):
        if path not in phash_cache:
            phash_cache[path] = compute_dhash(path)
        return phash_cache[path]

    for i, new_path in enumerate(new_image_paths):
        if progress_callback:
            progress_callback(i + 1, total, os.path.basename(new_path))

        new_embedding = get_image_embedding(new_path)
        if new_embedding is None:
            continue

        # Use filename for external images (they're not in the index directory)
        new_rel_path = os.path.basename(new_path)

        # Compute hash for new image once (if using hybrid)
        new_hash = get_phash(new_path) if use_hybrid else None

        for entry in index_entries:
            # Skip excluded pairs
            index_rel_path = index.get_relative_path(entry.path)
            if index.is_excluded(new_rel_path, index_rel_path, exclusion_type="comparison"):
                continue

            clip_sim = cosine_similarity(new_embedding, entry.embedding)

            if use_hybrid:
                # Pre-filter with CLIP
                if clip_sim < clip_prefilter:
                    continue

                # Compute perceptual hash similarity
                entry_hash = get_phash(entry.path)
                phash_sim = hash_similarity(new_hash, entry_hash)

                # Only include if perceptual hash passes threshold
                if phash_sim >= phash_threshold:
                    duplicates.append(DuplicateMatch(
                        image1_path=new_path,
                        image2_path=entry.path,
                        similarity=clip_sim,
                        phash_similarity=phash_sim
                    ))
            else:
                # Original CLIP-only behavior
                if clip_sim >= threshold:
                    duplicates.append(DuplicateMatch(
                        image1_path=new_path,
                        image2_path=entry.path,
                        similarity=clip_sim,
                        phash_similarity=0.0
                    ))

    # Sort by combined score (or CLIP similarity if not hybrid)
    if use_hybrid:
        duplicates.sort(key=lambda x: x.phash_similarity, reverse=True)
    else:
        duplicates.sort(key=lambda x: x.similarity, reverse=True)
    return duplicates
