"""Custom dialogs for displaying results and grouping."""

import os
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
from typing import List, Callable, Optional

from core.comparison import DuplicateMatch
from core.grouping import ImageGroup


class DuplicatesDialog(tk.Toplevel):
    """Dialog for displaying duplicate image results."""

    def __init__(
        self,
        parent,
        duplicates: List[DuplicateMatch],
        title: str = "Duplicate Results",
        index=None,
        exclusion_type: str = "duplicate",
        on_exclusions_changed: Optional[Callable[[], None]] = None
    ):
        super().__init__(parent)
        self.title(title)
        self.geometry("950x600")
        self.duplicates = list(duplicates)  # Make a copy so we can modify
        self.thumbnail_cache = {}
        self.index = index
        self.exclusion_type = exclusion_type  # "duplicate" or "comparison"
        self.on_exclusions_changed = on_exclusions_changed

        self._create_widgets()
        self._populate_results()

    def _create_widgets(self):
        # Button frame at bottom (pack first with side=BOTTOM so it stays visible)
        button_frame = ttk.Frame(self)
        button_frame.pack(side=tk.BOTTOM, pady=10)

        # Exclude button (only if index is provided)
        if self.index is not None:
            self.exclude_btn = ttk.Button(
                button_frame,
                text="Exclude Selected Pair",
                command=self._exclude_selected
            )
            self.exclude_btn.pack(side=tk.LEFT, padx=5)

        # Close button
        ttk.Button(button_frame, text="Close", command=self.destroy).pack(side=tk.LEFT, padx=5)

        # Preview frame (pack second from bottom with fixed height)
        preview_frame = ttk.LabelFrame(self, text="Preview", padding="10")
        preview_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)

        self.preview_frame_inner = ttk.Frame(preview_frame, height=160)
        self.preview_frame_inner.pack(fill=tk.X)
        self.preview_frame_inner.pack_propagate(False)  # Prevent resize from image content

        # Image labels for preview
        self.img1_label = ttk.Label(self.preview_frame_inner, text="Select a pair to preview")
        self.img1_label.pack(side=tk.LEFT, padx=20, expand=True)

        self.img2_label = ttk.Label(self.preview_frame_inner, text="")
        self.img2_label.pack(side=tk.RIGHT, padx=20, expand=True)

        # Main frame for treeview (fills remaining space)
        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Summary label
        self.summary_label = ttk.Label(
            main_frame,
            text=f"Found {len(self.duplicates)} duplicate pair(s)",
            font=('TkDefaultFont', 12, 'bold')
        )
        self.summary_label.pack(pady=(0, 10))

        # Tree frame
        tree_frame = ttk.Frame(main_frame)
        tree_frame.pack(fill=tk.BOTH, expand=True)

        # Create treeview for results
        columns = ('image1', 'image2', 'clip_sim', 'phash_sim')
        self.tree = ttk.Treeview(tree_frame, columns=columns, show='headings', height=15)

        self.tree.heading('image1', text='Image 1')
        self.tree.heading('image2', text='Image 2')
        self.tree.heading('clip_sim', text='CLIP')
        self.tree.heading('phash_sim', text='pHash')

        self.tree.column('image1', width=300)
        self.tree.column('image2', width=300)
        self.tree.column('clip_sim', width=70)
        self.tree.column('phash_sim', width=70)

        # Scrollbar
        scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)

        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Bind selection event
        self.tree.bind('<<TreeviewSelect>>', self._on_select)

        # Bind keyboard shortcuts (only if index provided for exclusions)
        if self.index is not None:
            self.tree.bind('<Delete>', lambda e: self._exclude_selected())
            self.tree.bind('<BackSpace>', lambda e: self._exclude_selected())

    def _populate_results(self):
        for dup in self.duplicates:
            phash_display = f"{dup.phash_similarity:.4f}" if dup.phash_similarity > 0 else "-"
            self.tree.insert('', tk.END, values=(
                dup.image1_name,
                dup.image2_name,
                f"{dup.similarity:.4f}",
                phash_display
            ))

    def _get_thumbnail(self, path: str, size: tuple = (150, 150)) -> Optional[ImageTk.PhotoImage]:
        if path in self.thumbnail_cache:
            return self.thumbnail_cache[path]

        try:
            img = Image.open(path)
            img.thumbnail(size, Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            self.thumbnail_cache[path] = photo
            return photo
        except Exception:
            return None

    def _on_select(self, event):
        selection = self.tree.selection()
        if not selection:
            return

        item = self.tree.item(selection[0])
        idx = self.tree.index(selection[0])

        if idx < len(self.duplicates):
            dup = self.duplicates[idx]

            # Update previews
            thumb1 = self._get_thumbnail(dup.image1_path)
            thumb2 = self._get_thumbnail(dup.image2_path)

            if thumb1:
                self.img1_label.configure(image=thumb1, text="")
            else:
                self.img1_label.configure(image="", text=dup.image1_name)

            if thumb2:
                self.img2_label.configure(image=thumb2, text="")
            else:
                self.img2_label.configure(image="", text=dup.image2_name)

    def _exclude_selected(self):
        """Exclude the selected pair from future results."""
        selection = self.tree.selection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a pair to exclude.")
            return

        item_id = selection[0]
        idx = self.tree.index(item_id)

        if idx < len(self.duplicates):
            dup = self.duplicates[idx]

            # Get relative paths for exclusion
            if self.exclusion_type == "comparison":
                # For comparison: image1 is external (use basename), image2 is in index
                rel_path1 = os.path.basename(dup.image1_path)
                rel_path2 = self.index.get_relative_path(dup.image2_path)
            else:
                # For duplicates within index: both are in index
                rel_path1 = self.index.get_relative_path(dup.image1_path)
                rel_path2 = self.index.get_relative_path(dup.image2_path)

            # Add exclusion to index
            self.index.add_exclusion(rel_path1, rel_path2, self.exclusion_type)

            # Remove from tree and list
            self.tree.delete(item_id)
            self.duplicates.pop(idx)

            # Select next item (or previous if at end)
            children = self.tree.get_children()
            if children:
                next_idx = min(idx, len(children) - 1)
                next_item = children[next_idx]
                self.tree.selection_set(next_item)
                self.tree.focus(next_item)

            # Notify that exclusions changed
            if self.on_exclusions_changed:
                self.on_exclusions_changed()

            # Update summary label
            self.summary_label.config(text=f"Found {len(self.duplicates)} duplicate pair(s)")


class GroupingDialog(tk.Toplevel):
    """Dialog for displaying and editing image groups."""

    def __init__(
        self,
        parent,
        groups: List[ImageGroup],
        on_rename: Optional[Callable[[List[ImageGroup]], None]] = None
    ):
        super().__init__(parent)
        self.title("Image Groups")
        self.geometry("800x600")
        self.groups = groups
        self.on_rename = on_rename
        self.thumbnail_cache = {}
        self.group_name_vars = []

        self._create_widgets()
        self._populate_groups()

    def _create_widgets(self):
        # Main frame with scrollable canvas
        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Summary
        summary = f"Found {len(self.groups)} group(s) with {sum(len(g.images) for g in self.groups)} images"
        ttk.Label(main_frame, text=summary, font=('TkDefaultFont', 12, 'bold')).pack(pady=(0, 10))

        # Canvas for scrolling
        canvas = tk.Canvas(main_frame)
        scrollbar = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, command=canvas.yview)
        self.scrollable_frame = ttk.Frame(canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Buttons frame
        button_frame = ttk.Frame(self)
        button_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Button(button_frame, text="Apply Rename", command=self._apply_rename).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Close", command=self.destroy).pack(side=tk.RIGHT, padx=5)

    def _get_thumbnail(self, path: str, size: tuple = (80, 80)) -> Optional[ImageTk.PhotoImage]:
        if path in self.thumbnail_cache:
            return self.thumbnail_cache[path]

        try:
            img = Image.open(path)
            img.thumbnail(size, Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            self.thumbnail_cache[path] = photo
            return photo
        except Exception:
            return None

    def _populate_groups(self):
        for i, group in enumerate(self.groups):
            group_frame = ttk.LabelFrame(
                self.scrollable_frame,
                text=f"Group {i+1}: {group.description} ({len(group.images)} images)",
                padding="10"
            )
            group_frame.pack(fill=tk.X, pady=5, padx=5)

            # Group name entry
            name_frame = ttk.Frame(group_frame)
            name_frame.pack(fill=tk.X, pady=(0, 5))

            ttk.Label(name_frame, text="Group Name:").pack(side=tk.LEFT)

            name_var = tk.StringVar(value=group.suggested_name)
            self.group_name_vars.append(name_var)

            name_entry = ttk.Entry(name_frame, textvariable=name_var, width=30)
            name_entry.pack(side=tk.LEFT, padx=5)

            ttk.Label(name_frame, text=f"(confidence: {group.confidence:.2f})").pack(side=tk.LEFT)

            # Thumbnail strip
            thumb_frame = ttk.Frame(group_frame)
            thumb_frame.pack(fill=tk.X)

            # Show up to 8 thumbnails
            for j, entry in enumerate(group.images[:8]):
                thumb = self._get_thumbnail(entry.path)
                if thumb:
                    lbl = ttk.Label(thumb_frame, image=thumb)
                    lbl.pack(side=tk.LEFT, padx=2)

            if len(group.images) > 8:
                ttk.Label(thumb_frame, text=f"+{len(group.images) - 8} more").pack(side=tk.LEFT, padx=5)

    def _apply_rename(self):
        # Update group names from entries
        for group, name_var in zip(self.groups, self.group_name_vars):
            group.suggested_name = name_var.get()

        if self.on_rename:
            if messagebox.askyesno(
                "Confirm Rename",
                "This will rename files on disk. Continue?"
            ):
                self.on_rename(self.groups)
                messagebox.showinfo("Complete", "Files have been renamed.")
                self.destroy()


class ProgressDialog(tk.Toplevel):
    """Dialog for showing progress during long operations."""

    def __init__(self, parent, title: str = "Processing..."):
        super().__init__(parent)
        self.title(title)
        self.geometry("400x150")
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()

        self._destroyed = False
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self._create_widgets()

    def _on_close(self):
        """Handle window close button."""
        self._destroyed = True
        self.destroy()

    def destroy(self):
        """Override destroy to set flag."""
        self._destroyed = True
        super().destroy()

    def _create_widgets(self):
        main_frame = ttk.Frame(self, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)

        self.status_label = ttk.Label(main_frame, text="Initializing...")
        self.status_label.pack(pady=(0, 10))

        self.progress = ttk.Progressbar(main_frame, length=350, mode='determinate')
        self.progress.pack(pady=5)

        self.detail_label = ttk.Label(main_frame, text="")
        self.detail_label.pack(pady=5)

    def update_progress(self, current: int, total: int, detail: str = ""):
        """Update progress bar and labels."""
        if self._destroyed:
            return
        try:
            if total > 0:
                self.progress['value'] = (current / total) * 100
            self.status_label['text'] = f"Processing {current} of {total}"
            self.detail_label['text'] = detail
            self.update()
        except tk.TclError:
            # Widget was destroyed
            self._destroyed = True

    def set_status(self, text: str):
        """Set the status text."""
        if self._destroyed:
            return
        try:
            self.status_label['text'] = text
            self.update()
        except tk.TclError:
            # Widget was destroyed
            self._destroyed = True


class ExclusionsDialog(tk.Toplevel):
    """Dialog for managing exclusion lists."""

    def __init__(
        self,
        parent,
        index,
        on_exclusions_changed: Optional[Callable[[], None]] = None
    ):
        super().__init__(parent)
        self.title("Manage Exclusions")
        self.geometry("800x650")
        self.index = index
        self.on_exclusions_changed = on_exclusions_changed
        self._modified = False
        self.thumbnail_cache = {}

        self._create_widgets()
        self._populate_lists()

    def _create_widgets(self):
        # Bottom buttons (pack first with side=BOTTOM)
        button_frame = ttk.Frame(self)
        button_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=(0, 10))
        ttk.Button(button_frame, text="Close", command=self._on_close).pack(side=tk.RIGHT, padx=5)

        # Preview frame (pack second from bottom)
        preview_outer_frame = ttk.LabelFrame(self, text="Preview", padding="10")
        preview_outer_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)

        self.preview_frame = ttk.Frame(preview_outer_frame, height=160)
        self.preview_frame.pack(fill=tk.X)
        self.preview_frame.pack_propagate(False)

        # Image labels for preview
        self.img1_frame = ttk.Frame(self.preview_frame)
        self.img1_frame.pack(side=tk.LEFT, padx=20, expand=True)
        self.img1_label = ttk.Label(self.img1_frame, text="Select a pair to preview")
        self.img1_label.pack()
        self.img1_path_label = ttk.Label(self.img1_frame, text="", font=('TkDefaultFont', 8))
        self.img1_path_label.pack()

        self.img2_frame = ttk.Frame(self.preview_frame)
        self.img2_frame.pack(side=tk.RIGHT, padx=20, expand=True)
        self.img2_label = ttk.Label(self.img2_frame, text="")
        self.img2_label.pack()
        self.img2_path_label = ttk.Label(self.img2_frame, text="", font=('TkDefaultFont', 8))
        self.img2_path_label.pack()

        # Create notebook for tabs
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Duplicate exclusions tab
        self.dup_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.dup_frame, text="Duplicate Exclusions")
        self._create_list_frame(self.dup_frame, "duplicate")

        # Comparison exclusions tab
        self.cmp_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.cmp_frame, text="Comparison Exclusions")
        self._create_list_frame(self.cmp_frame, "comparison")

    def _create_list_frame(self, parent, exclusion_type: str):
        """Create a list frame for exclusions."""
        # Info label
        if exclusion_type == "duplicate":
            info_text = "Pairs excluded from 'Find Duplicates in Index' results:"
        else:
            info_text = "Pairs excluded from 'Compare Images' results (external image path not available):"

        ttk.Label(parent, text=info_text).pack(anchor=tk.W, padx=5, pady=5)

        # Tree frame
        tree_frame = ttk.Frame(parent)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Create treeview
        columns = ('image1', 'image2')
        tree = ttk.Treeview(tree_frame, columns=columns, show='headings', height=10)

        tree.heading('image1', text='Image 1')
        tree.heading('image2', text='Image 2')

        tree.column('image1', width=350)
        tree.column('image2', width=350)

        # Scrollbar
        scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)

        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Bind selection event
        tree.bind('<<TreeviewSelect>>', lambda e, t=exclusion_type: self._on_select(e, t))

        # Store reference
        if exclusion_type == "duplicate":
            self.dup_tree = tree
        else:
            self.cmp_tree = tree

        # Button frame for this list
        list_button_frame = ttk.Frame(parent)
        list_button_frame.pack(fill=tk.X, padx=5, pady=5)

        remove_btn = ttk.Button(
            list_button_frame,
            text="Remove Selected",
            command=lambda: self._remove_selected(exclusion_type)
        )
        remove_btn.pack(side=tk.LEFT, padx=5)

        clear_btn = ttk.Button(
            list_button_frame,
            text="Clear All",
            command=lambda: self._clear_all(exclusion_type)
        )
        clear_btn.pack(side=tk.LEFT, padx=5)

        # Count label
        count_label = ttk.Label(list_button_frame, text="")
        count_label.pack(side=tk.RIGHT, padx=5)

        if exclusion_type == "duplicate":
            self.dup_count_label = count_label
        else:
            self.cmp_count_label = count_label

    def _get_thumbnail(self, path: str, size: tuple = (130, 130)) -> Optional[ImageTk.PhotoImage]:
        """Get thumbnail for an image path."""
        if path in self.thumbnail_cache:
            return self.thumbnail_cache[path]

        try:
            img = Image.open(path)
            img.thumbnail(size, Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            self.thumbnail_cache[path] = photo
            return photo
        except Exception:
            return None

    def _get_absolute_path(self, relative_path: str) -> Optional[str]:
        """Convert relative path to absolute path using index source directory."""
        if not self.index.source_directory:
            return None
        abs_path = os.path.join(self.index.source_directory, relative_path)
        if os.path.exists(abs_path):
            return abs_path
        return None

    def _on_select(self, event, exclusion_type: str):
        """Handle selection in treeview."""
        tree = self.dup_tree if exclusion_type == "duplicate" else self.cmp_tree
        selection = tree.selection()

        if not selection:
            return

        values = tree.item(selection[0])['values']
        if len(values) != 2:
            return

        rel_path1, rel_path2 = str(values[0]), str(values[1])

        # For comparison exclusions, first image is external (just filename, no full path)
        if exclusion_type == "comparison":
            # First image is external - can't preview
            self.img1_label.configure(image="", text="(External image)")
            self.img1_path_label.configure(text=rel_path1)

            # Second image is in index
            abs_path2 = self._get_absolute_path(rel_path2)
            if abs_path2:
                thumb2 = self._get_thumbnail(abs_path2)
                if thumb2:
                    self.img2_label.configure(image=thumb2, text="")
                else:
                    self.img2_label.configure(image="", text="(Load failed)")
            else:
                self.img2_label.configure(image="", text="(File not found)")
            self.img2_path_label.configure(text=rel_path2)
        else:
            # Both images are in index
            abs_path1 = self._get_absolute_path(rel_path1)
            abs_path2 = self._get_absolute_path(rel_path2)

            if abs_path1:
                thumb1 = self._get_thumbnail(abs_path1)
                if thumb1:
                    self.img1_label.configure(image=thumb1, text="")
                else:
                    self.img1_label.configure(image="", text="(Load failed)")
            else:
                self.img1_label.configure(image="", text="(File not found)")
            self.img1_path_label.configure(text=rel_path1)

            if abs_path2:
                thumb2 = self._get_thumbnail(abs_path2)
                if thumb2:
                    self.img2_label.configure(image=thumb2, text="")
                else:
                    self.img2_label.configure(image="", text="(Load failed)")
            else:
                self.img2_label.configure(image="", text="(File not found)")
            self.img2_path_label.configure(text=rel_path2)

    def _populate_lists(self):
        """Populate both exclusion lists."""
        self._populate_tree(self.dup_tree, self.index.duplicate_exclusions, self.dup_count_label)
        self._populate_tree(self.cmp_tree, self.index.comparison_exclusions, self.cmp_count_label)

    def _populate_tree(self, tree: ttk.Treeview, exclusions: set, count_label: ttk.Label):
        """Populate a single tree with exclusions."""
        # Clear existing
        for item in tree.get_children():
            tree.delete(item)

        # Add exclusions
        for pair in sorted(exclusions, key=lambda p: tuple(sorted(p))):
            items = sorted(pair)
            if len(items) == 2:
                tree.insert('', tk.END, values=(items[0], items[1]))

        # Update count
        count_label.config(text=f"{len(exclusions)} exclusion(s)")

    def _remove_selected(self, exclusion_type: str):
        """Remove selected exclusion."""
        tree = self.dup_tree if exclusion_type == "duplicate" else self.cmp_tree
        count_label = self.dup_count_label if exclusion_type == "duplicate" else self.cmp_count_label

        selection = tree.selection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select an exclusion to remove.")
            return

        for item_id in selection:
            values = tree.item(item_id)['values']
            if len(values) == 2:
                self.index.remove_exclusion(str(values[0]), str(values[1]), exclusion_type)
                tree.delete(item_id)

        self._modified = True

        # Clear preview
        self.img1_label.configure(image="", text="Select a pair to preview")
        self.img1_path_label.configure(text="")
        self.img2_label.configure(image="", text="")
        self.img2_path_label.configure(text="")

        # Update count
        exclusions = self.index.duplicate_exclusions if exclusion_type == "duplicate" else self.index.comparison_exclusions
        count_label.config(text=f"{len(exclusions)} exclusion(s)")

    def _clear_all(self, exclusion_type: str):
        """Clear all exclusions of a type."""
        exclusions = self.index.duplicate_exclusions if exclusion_type == "duplicate" else self.index.comparison_exclusions

        if not exclusions:
            messagebox.showinfo("Empty", "No exclusions to clear.")
            return

        type_name = "duplicate" if exclusion_type == "duplicate" else "comparison"
        if not messagebox.askyesno(
            "Confirm Clear",
            f"Remove all {len(exclusions)} {type_name} exclusion(s)?"
        ):
            return

        exclusions.clear()
        self.index.modified_at = __import__('datetime').datetime.now()
        self._modified = True

        # Clear preview
        self.img1_label.configure(image="", text="Select a pair to preview")
        self.img1_path_label.configure(text="")
        self.img2_label.configure(image="", text="")
        self.img2_path_label.configure(text="")

        # Refresh the list
        tree = self.dup_tree if exclusion_type == "duplicate" else self.cmp_tree
        count_label = self.dup_count_label if exclusion_type == "duplicate" else self.cmp_count_label
        self._populate_tree(tree, exclusions, count_label)

    def _on_close(self):
        """Handle close button."""
        if self._modified and self.on_exclusions_changed:
            self.on_exclusions_changed()
        self.destroy()
