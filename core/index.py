"""Index creation, persistence, and loading for image embeddings."""

import os
import pickle
import hashlib
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional, Callable, Tuple
from dataclasses import dataclass, field

from .embeddings import get_image_embedding


# Supported image extensions
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.tif'}


def compute_file_hash(filepath: str) -> Optional[str]:
    """Compute MD5 hash of a file."""
    try:
        hasher = hashlib.md5()
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception:
        return None


@dataclass
class ImageEntry:
    """Represents a single indexed image."""
    filename: str
    path: str
    embedding: np.ndarray
    indexed_at: datetime = field(default_factory=datetime.now)
    file_hash: Optional[str] = None


class ImageIndex:
    """
    Manages a collection of image embeddings for duplicate detection.

    Supports creation from directory, persistence to/from pickle files,
    and incremental updates.
    """

    def __init__(self):
        self.entries: Dict[str, ImageEntry] = {}  # path -> ImageEntry
        self.source_directory: Optional[str] = None
        self.created_at: Optional[datetime] = None
        self.modified_at: Optional[datetime] = None
        # Separate exclusion lists for different comparison types
        # Each stores frozenset({relative_path1, relative_path2}) pairs
        self.comparison_exclusions: set = set()  # For "Compare Images" (external vs index)
        self.duplicate_exclusions: set = set()   # For "Find Duplicates in Index" (within index)

    @property
    def count(self) -> int:
        """Number of images in the index."""
        return len(self.entries)

    def get_relative_path(self, absolute_path: str) -> str:
        """Get relative path from absolute path based on source directory."""
        if self.source_directory:
            try:
                return os.path.relpath(absolute_path, self.source_directory)
            except ValueError:
                # Different drives on Windows
                return os.path.basename(absolute_path)
        return os.path.basename(absolute_path)

    def add_exclusion(self, path1: str, path2: str, exclusion_type: str = "duplicate") -> None:
        """
        Add an exclusion pair (by relative path).

        Args:
            path1: Relative path of first image
            path2: Relative path of second image
            exclusion_type: "duplicate" for within-index, "comparison" for external comparison
        """
        pair = frozenset({path1, path2})
        if exclusion_type == "comparison":
            self.comparison_exclusions.add(pair)
        else:
            self.duplicate_exclusions.add(pair)
        self.modified_at = datetime.now()

    def remove_exclusion(self, path1: str, path2: str, exclusion_type: str = "duplicate") -> bool:
        """Remove an exclusion pair. Returns True if it existed."""
        pair = frozenset({path1, path2})
        exclusions = self.comparison_exclusions if exclusion_type == "comparison" else self.duplicate_exclusions
        if pair in exclusions:
            exclusions.discard(pair)
            self.modified_at = datetime.now()
            return True
        return False

    def is_excluded(self, path1: str, path2: str, exclusion_type: str = "duplicate") -> bool:
        """Check if a pair is excluded."""
        pair = frozenset({path1, path2})
        exclusions = self.comparison_exclusions if exclusion_type == "comparison" else self.duplicate_exclusions
        return pair in exclusions

    def get_all_entries(self) -> List[ImageEntry]:
        """Get all image entries in the index."""
        return list(self.entries.values())

    def get_all_embeddings(self) -> List[tuple]:
        """Get list of (path, embedding) tuples."""
        return [(e.path, e.embedding) for e in self.entries.values()]

    @classmethod
    def create_from_directory(
        cls,
        directory_path: str,
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> "ImageIndex":
        """
        Create a new index by scanning and embedding all images in a directory.

        Args:
            directory_path: Path to directory containing images
            progress_callback: Optional callback(current, total, filename) for progress updates

        Returns:
            New ImageIndex instance
        """
        index = cls()
        index.source_directory = os.path.abspath(directory_path)
        index.created_at = datetime.now()
        index.modified_at = index.created_at

        # Find all image files
        image_files = []
        for fname in os.listdir(directory_path):
            ext = os.path.splitext(fname)[1].lower()
            if ext in IMAGE_EXTENSIONS:
                image_files.append(fname)

        total = len(image_files)
        for i, fname in enumerate(image_files):
            full_path = os.path.join(directory_path, fname)

            if progress_callback:
                progress_callback(i + 1, total, fname)

            if os.path.isfile(full_path):
                embedding = get_image_embedding(full_path)
                if embedding is not None:
                    entry = ImageEntry(
                        filename=fname,
                        path=full_path,
                        embedding=embedding,
                        file_hash=compute_file_hash(full_path)
                    )
                    index.entries[full_path] = entry

        return index

    def add_images(
        self,
        paths: List[str],
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> int:
        """
        Incrementally add new images to the index.

        Args:
            paths: List of image file paths to add
            progress_callback: Optional callback(current, total, filename) for progress updates

        Returns:
            Number of images successfully added
        """
        added = 0
        total = len(paths)

        for i, path in enumerate(paths):
            full_path = os.path.abspath(path)

            if progress_callback:
                progress_callback(i + 1, total, os.path.basename(path))

            # Skip if already indexed
            if full_path in self.entries:
                continue

            if os.path.isfile(full_path):
                embedding = get_image_embedding(full_path)
                if embedding is not None:
                    entry = ImageEntry(
                        filename=os.path.basename(full_path),
                        path=full_path,
                        embedding=embedding,
                        file_hash=compute_file_hash(full_path)
                    )
                    self.entries[full_path] = entry
                    added += 1

        if added > 0:
            self.modified_at = datetime.now()

        return added

    def remove_missing(
        self,
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> Tuple[int, int]:
        """
        Remove entries for missing images and re-embed changed images.

        Args:
            progress_callback: Optional callback(current, total, status) for progress updates

        Returns:
            Tuple of (removed_count, updated_count)
        """
        missing = []
        changed = []

        entries_list = list(self.entries.items())
        total = len(entries_list)

        # Phase 1: Check for missing and changed files
        for i, (path, entry) in enumerate(entries_list):
            if progress_callback:
                progress_callback(i + 1, total, f"Checking: {entry.filename}")

            if not os.path.exists(path):
                missing.append(path)
            else:
                # Check if content changed
                current_hash = compute_file_hash(path)
                if current_hash and entry.file_hash != current_hash:
                    changed.append((path, current_hash))

        # Remove missing entries
        for path in missing:
            del self.entries[path]

        # Phase 2: Re-embed changed files
        total_changed = len(changed)
        for i, (path, new_hash) in enumerate(changed):
            if progress_callback:
                progress_callback(i + 1, total_changed, f"Re-indexing: {os.path.basename(path)}")

            embedding = get_image_embedding(path)
            if embedding is not None:
                self.entries[path].embedding = embedding
                self.entries[path].file_hash = new_hash
                self.entries[path].indexed_at = datetime.now()

        if missing or changed:
            self.modified_at = datetime.now()

        return len(missing), len(changed)

    def save(self, filepath: str) -> None:
        """
        Save the index to a pickle file.

        Args:
            filepath: Path to save the index file
        """
        data = {
            'entries': self.entries,
            'source_directory': self.source_directory,
            'created_at': self.created_at,
            'modified_at': datetime.now(),
            'comparison_exclusions': self.comparison_exclusions,
            'duplicate_exclusions': self.duplicate_exclusions,
            'version': 3
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, filepath: str) -> "ImageIndex":
        """
        Load an index from a pickle file.

        Args:
            filepath: Path to the index file

        Returns:
            Loaded ImageIndex instance
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        index = cls()
        index.entries = data['entries']
        index.source_directory = data.get('source_directory')
        index.created_at = data.get('created_at')
        index.modified_at = data.get('modified_at')

        # Handle migration from old format (version 2 had single 'exclusions')
        if 'exclusions' in data and 'duplicate_exclusions' not in data:
            # Migrate old exclusions to duplicate_exclusions
            index.duplicate_exclusions = data.get('exclusions', set())
            index.comparison_exclusions = set()
        else:
            index.comparison_exclusions = data.get('comparison_exclusions', set())
            index.duplicate_exclusions = data.get('duplicate_exclusions', set())

        return index

    def __repr__(self) -> str:
        return f"ImageIndex(count={self.count}, source={self.source_directory})"
