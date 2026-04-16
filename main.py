#!/usr/bin/env python3
"""
Image Duplicate Detector - Desktop Application

A Tkinter-based GUI application for detecting duplicate images using
OpenAI's CLIP model for visual similarity comparison.

Features:
- Create and save image indices
- Find duplicates within an index
- Compare new images against an existing index
- Group images by subject using clustering
"""

import sys


def main():
    """Launch the Image Duplicate Detector GUI application."""
    # Import here to allow for faster --help response if needed
    from gui.app import DuplicateDetectorApp

    app = DuplicateDetectorApp()
    app.mainloop()


if __name__ == "__main__":
    main()
