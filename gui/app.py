"""Main Tkinter application window for Image Duplicate Detector."""

import os
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import Optional

from core.index import ImageIndex
from core.comparison import find_duplicates_in_index, find_duplicates_against_index
from core.grouping import group_by_subject, rename_images_by_group
from gui.dialogs import DuplicatesDialog, GroupingDialog, ProgressDialog, ExclusionsDialog


class DuplicateDetectorApp(tk.Tk):
    """Main application window for the Image Duplicate Detector."""

    def __init__(self):
        super().__init__()

        self.title("Image Duplicate Detector")
        self.geometry("600x400")

        self.index: Optional[ImageIndex] = None
        self.index_path: Optional[str] = None
        self._shutting_down = False

        self._create_menu()
        self._create_widgets()
        self._update_status()

        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _on_close(self):
        """Handle window close."""
        self._shutting_down = True
        self.destroy()

    def _safe_after(self, callback):
        """Safely schedule a callback on the main thread."""
        if self._shutting_down:
            return

        def safe_callback():
            if self._shutting_down:
                return
            try:
                callback()
            except tk.TclError:
                # Widget was destroyed
                pass

        try:
            self.after(0, safe_callback)
        except RuntimeError:
            # Main thread is not in main loop
            pass

    def _create_menu(self):
        """Create the menu bar."""
        menubar = tk.Menu(self)
        self.config(menu=menubar)

        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)

        file_menu.add_command(label="Open Index...", command=self._load_index)
        file_menu.add_command(label="Save Index...", command=self._save_index)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.quit)

    def _create_widgets(self):
        """Create the main window widgets."""
        # Main frame
        main_frame = ttk.Frame(self, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Title
        title_label = ttk.Label(
            main_frame,
            text="Image Duplicate Detector",
            font=('TkDefaultFont', 16, 'bold')
        )
        title_label.pack(pady=(0, 20))

        # Button panel
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=10)

        # Create buttons with consistent styling
        button_width = 20

        self.create_btn = ttk.Button(
            button_frame,
            text="Create Index",
            width=button_width,
            command=self._create_index
        )
        self.create_btn.pack(pady=5)

        self.load_btn = ttk.Button(
            button_frame,
            text="Load Index",
            width=button_width,
            command=self._load_index
        )
        self.load_btn.pack(pady=5)

        self.refresh_btn = ttk.Button(
            button_frame,
            text="Refresh Index",
            width=button_width,
            command=self._refresh_index
        )
        self.refresh_btn.pack(pady=5)

        self.compare_btn = ttk.Button(
            button_frame,
            text="Compare Images",
            width=button_width,
            command=self._compare_images
        )
        self.compare_btn.pack(pady=5)

        self.find_dups_btn = ttk.Button(
            button_frame,
            text="Find Duplicates in Index",
            width=button_width,
            command=self._find_duplicates_in_index
        )
        self.find_dups_btn.pack(pady=5)

        self.group_btn = ttk.Button(
            button_frame,
            text="Group by Subject",
            width=button_width,
            command=self._group_by_subject
        )
        self.group_btn.pack(pady=5)

        self.exclusions_btn = ttk.Button(
            button_frame,
            text="Manage Exclusions",
            width=button_width,
            command=self._manage_exclusions
        )
        self.exclusions_btn.pack(pady=5)

        # Settings frame
        settings_frame = ttk.LabelFrame(main_frame, text="Comparison Settings", padding="5")
        settings_frame.pack(fill=tk.X, pady=10)

        # Hybrid mode toggle
        self.hybrid_var = tk.BooleanVar(value=True)
        hybrid_check = ttk.Checkbutton(
            settings_frame,
            text="Hybrid Mode (CLIP + Perceptual Hash)",
            variable=self.hybrid_var
        )
        hybrid_check.pack(anchor=tk.W)

        # Thresholds row
        threshold_frame = ttk.Frame(settings_frame)
        threshold_frame.pack(fill=tk.X, pady=5)

        ttk.Label(threshold_frame, text="CLIP Pre-filter:").pack(side=tk.LEFT)
        self.clip_threshold_var = tk.StringVar(value="0.85")
        ttk.Entry(threshold_frame, textvariable=self.clip_threshold_var, width=6).pack(side=tk.LEFT, padx=2)

        ttk.Label(threshold_frame, text="  pHash Threshold:").pack(side=tk.LEFT)
        self.phash_threshold_var = tk.StringVar(value="0.90")
        ttk.Entry(threshold_frame, textvariable=self.phash_threshold_var, width=6).pack(side=tk.LEFT, padx=2)

        # Legacy threshold (used when hybrid mode is off)
        ttk.Label(threshold_frame, text="  CLIP-only:").pack(side=tk.LEFT)
        self.threshold_var = tk.StringVar(value="0.995")
        ttk.Entry(threshold_frame, textvariable=self.threshold_var, width=6).pack(side=tk.LEFT, padx=2)

        # Status area
        status_frame = ttk.LabelFrame(main_frame, text="Status", padding="10")
        status_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        self.status_text = tk.Text(status_frame, height=8, state=tk.DISABLED)
        self.status_text.pack(fill=tk.BOTH, expand=True)

    def _update_status(self):
        """Update the status display."""
        self.status_text.config(state=tk.NORMAL)
        self.status_text.delete(1.0, tk.END)

        if self.index is None:
            self.status_text.insert(tk.END, "No index loaded.\n\n")
            self.status_text.insert(tk.END, "Click 'Create Index' to scan a directory,\n")
            self.status_text.insert(tk.END, "or 'Load Index' to load an existing index.")
        else:
            self.status_text.insert(tk.END, f"Index loaded: {self.index.count} images\n")
            if self.index.source_directory:
                self.status_text.insert(tk.END, f"Source: {self.index.source_directory}\n")
            if self.index_path:
                self.status_text.insert(tk.END, f"File: {self.index_path}\n")
            if self.index.created_at:
                self.status_text.insert(tk.END, f"Created: {self.index.created_at.strftime('%Y-%m-%d %H:%M')}\n")
            if self.index.modified_at:
                self.status_text.insert(tk.END, f"Modified: {self.index.modified_at.strftime('%Y-%m-%d %H:%M')}\n")
            # Show exclusion counts
            dup_excl = len(self.index.duplicate_exclusions)
            cmp_excl = len(self.index.comparison_exclusions)
            if dup_excl > 0 or cmp_excl > 0:
                self.status_text.insert(tk.END, f"Exclusions: {dup_excl} duplicate, {cmp_excl} comparison\n")

        self.status_text.config(state=tk.DISABLED)

    def _get_threshold(self) -> float:
        """Get the current similarity threshold value."""
        try:
            threshold = float(self.threshold_var.get())
            return max(0.0, min(1.0, threshold))
        except ValueError:
            return 0.995

    def _get_clip_prefilter(self) -> float:
        """Get the CLIP pre-filter threshold for hybrid mode."""
        try:
            return max(0.0, min(1.0, float(self.clip_threshold_var.get())))
        except ValueError:
            return 0.85

    def _get_phash_threshold(self) -> float:
        """Get the perceptual hash threshold for hybrid mode."""
        try:
            return max(0.0, min(1.0, float(self.phash_threshold_var.get())))
        except ValueError:
            return 0.90

    def _is_hybrid_mode(self) -> bool:
        """Check if hybrid mode is enabled."""
        return self.hybrid_var.get()

    def _create_index(self):
        """Create a new index from a directory."""
        directory = filedialog.askdirectory(title="Select Directory to Index")
        if not directory:
            return

        progress = ProgressDialog(self, "Creating Index...")

        def do_create():
            last_update = [0]

            def callback(current, total, filename):
                # Throttle updates to every 10 files or at completion
                if current - last_update[0] >= 10 or current == total:
                    last_update[0] = current
                    self._safe_after(lambda c=current, t=total, f=filename: progress.update_progress(c, t, f))

            try:
                index = ImageIndex.create_from_directory(directory, callback)
                self._safe_after(lambda: self._on_index_created(index, progress))
            except Exception as e:
                self._safe_after(lambda: self._on_error(str(e), progress))

        thread = threading.Thread(target=do_create)
        thread.start()

    def _on_index_created(self, index: ImageIndex, progress: ProgressDialog):
        """Handle index creation completion."""
        if self._shutting_down:
            return
        if not progress._destroyed:
            progress.destroy()
        self.index = index
        self.index_path = None
        self._update_status()

        # Prompt to save
        if messagebox.askyesno("Index Created", f"Index created with {index.count} images.\nSave to file?"):
            self._save_index()

    def _on_error(self, message: str, progress: Optional[ProgressDialog] = None):
        """Handle an error during processing."""
        if self._shutting_down:
            return
        if progress and not progress._destroyed:
            progress.destroy()
        messagebox.showerror("Error", message)

    def _save_index(self):
        """Save the current index to a file."""
        if self.index is None:
            messagebox.showwarning("No Index", "No index to save. Create or load an index first.")
            return

        filepath = filedialog.asksaveasfilename(
            title="Save Index",
            defaultextension=".pkl",
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")]
        )
        if not filepath:
            return

        try:
            self.index.save(filepath)
            self.index_path = filepath
            self._update_status()
            messagebox.showinfo("Saved", f"Index saved to {filepath}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save index: {e}")

    def _load_index(self):
        """Load an index from a file."""
        filepath = filedialog.askopenfilename(
            title="Load Index",
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")]
        )
        if not filepath:
            return

        progress = ProgressDialog(self, "Loading Index...")
        progress.set_status("Loading index file...")

        def do_load():
            try:
                index = ImageIndex.load(filepath)
                self._safe_after(lambda: self._on_index_loaded(index, filepath, progress))
            except Exception as e:
                self._safe_after(lambda: self._on_error(str(e), progress))

        thread = threading.Thread(target=do_load)
        thread.start()

    def _on_index_loaded(self, index: ImageIndex, filepath: str, progress: ProgressDialog):
        """Handle index load completion."""
        if self._shutting_down:
            return
        if not progress._destroyed:
            progress.destroy()
        self.index = index
        self.index_path = filepath
        self._update_status()
        messagebox.showinfo("Loaded", f"Index loaded: {index.count} images")

    def _refresh_index(self):
        """Re-scan directory and rebuild index, preserving exclusions."""
        if self.index is None:
            messagebox.showwarning("No Index", "Load or create an index first.")
            return

        if not self.index.source_directory:
            messagebox.showwarning("No Source", "Index has no source directory. Cannot refresh.")
            return

        if not os.path.isdir(self.index.source_directory):
            messagebox.showerror("Directory Not Found", f"Source directory no longer exists:\n{self.index.source_directory}")
            return

        # Save both exclusion sets to restore after
        saved_comparison_exclusions = self.index.comparison_exclusions.copy()
        saved_duplicate_exclusions = self.index.duplicate_exclusions.copy()
        old_count = self.index.count
        directory = self.index.source_directory

        progress = ProgressDialog(self, "Refreshing Index...")

        def do_refresh():
            last_update = [0]

            def callback(current, total, filename):
                if current - last_update[0] >= 10 or current == total:
                    last_update[0] = current
                    self._safe_after(lambda c=current, t=total, f=filename: progress.update_progress(c, t, f))

            try:
                new_index = ImageIndex.create_from_directory(directory, callback)
                new_index.comparison_exclusions = saved_comparison_exclusions
                new_index.duplicate_exclusions = saved_duplicate_exclusions
                self._safe_after(lambda: self._on_refresh_complete(new_index, old_count, progress))
            except Exception as e:
                self._safe_after(lambda: self._on_error(str(e), progress))

        thread = threading.Thread(target=do_refresh)
        thread.start()

    def _on_refresh_complete(self, new_index: ImageIndex, old_count: int, progress: ProgressDialog):
        """Handle refresh completion."""
        if self._shutting_down:
            return
        if not progress._destroyed:
            progress.destroy()

        self.index = new_index

        # Save if we have a path
        if self.index_path:
            try:
                self.index.save(self.index_path)
            except Exception as e:
                messagebox.showerror("Save Error", f"Failed to save: {e}")
                return

        self._update_status()
        messagebox.showinfo("Refreshed", f"Index refreshed.\nPrevious: {old_count} images\nNow: {self.index.count} images")

    def _compare_images(self):
        """Compare selected images against the loaded index."""
        if self.index is None:
            messagebox.showwarning("No Index", "Load or create an index first.")
            return

        filepaths = filedialog.askopenfilenames(
            title="Select Images to Compare",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.gif *.bmp *.webp"),
                ("All files", "*.*")
            ]
        )
        if not filepaths:
            return

        progress = ProgressDialog(self, "Comparing Images...")
        threshold = self._get_threshold()
        use_hybrid = self._is_hybrid_mode()
        clip_prefilter = self._get_clip_prefilter()
        phash_threshold = self._get_phash_threshold()

        def do_compare():
            last_update = [0]

            def callback(current, total, filename):
                # Throttle updates to every 100 comparisons or at completion
                if current - last_update[0] >= 100 or current == total:
                    last_update[0] = current
                    self._safe_after(lambda c=current, t=total, f=filename: progress.update_progress(c, t, f))

            try:
                duplicates = find_duplicates_against_index(
                    list(filepaths),
                    self.index,
                    threshold,
                    callback,
                    use_hybrid=use_hybrid,
                    clip_prefilter=clip_prefilter,
                    phash_threshold=phash_threshold
                )
                self._safe_after(lambda: self._on_comparison_complete(duplicates, progress))
            except Exception as e:
                self._safe_after(lambda: self._on_error(str(e), progress))

        thread = threading.Thread(target=do_compare)
        thread.start()

    def _on_comparison_complete(self, duplicates, progress: ProgressDialog):
        """Handle comparison completion."""
        if self._shutting_down:
            return
        if not progress._destroyed:
            progress.destroy()

        if not duplicates:
            messagebox.showinfo("No Duplicates", "No duplicates found above the threshold.")
        else:
            DuplicatesDialog(
                self,
                duplicates,
                "Comparison Results",
                index=self.index,
                exclusion_type="comparison",
                on_exclusions_changed=self._on_exclusions_changed
            )

    def _find_duplicates_in_index(self):
        """Find duplicates within the loaded index."""
        if self.index is None:
            messagebox.showwarning("No Index", "Load or create an index first.")
            return

        progress = ProgressDialog(self, "Finding Duplicates...")
        threshold = self._get_threshold()
        use_hybrid = self._is_hybrid_mode()
        clip_prefilter = self._get_clip_prefilter()
        phash_threshold = self._get_phash_threshold()

        def do_find():
            last_update = [0]

            def callback(current, total):
                # Throttle updates to every 1000 comparisons or at completion
                if current - last_update[0] >= 1000 or current == total:
                    last_update[0] = current
                    self._safe_after(lambda c=current, t=total: progress.update_progress(c, t, ""))

            try:
                duplicates = find_duplicates_in_index(
                    self.index,
                    threshold,
                    callback,
                    use_hybrid=use_hybrid,
                    clip_prefilter=clip_prefilter,
                    phash_threshold=phash_threshold
                )
                self._safe_after(lambda: self._on_duplicates_found(duplicates, progress))
            except Exception as e:
                self._safe_after(lambda: self._on_error(str(e), progress))

        thread = threading.Thread(target=do_find)
        thread.start()

    def _on_duplicates_found(self, duplicates, progress: ProgressDialog):
        """Handle duplicate finding completion."""
        if self._shutting_down:
            return
        if not progress._destroyed:
            progress.destroy()

        if not duplicates:
            messagebox.showinfo("No Duplicates", "No duplicates found above the threshold.")
        else:
            DuplicatesDialog(
                self,
                duplicates,
                "Duplicates in Index",
                index=self.index,
                exclusion_type="duplicate",
                on_exclusions_changed=self._on_exclusions_changed
            )

    def _on_exclusions_changed(self):
        """Handle when exclusions are modified."""
        # Auto-save if we have a path, otherwise prompt
        if self.index_path:
            try:
                self.index.save(self.index_path)
                self._update_status()
            except Exception as e:
                messagebox.showerror("Save Error", f"Failed to save exclusions: {e}")
        else:
            if messagebox.askyesno("Save Index", "Exclusions modified. Save index to preserve changes?"):
                self._save_index()
            self._update_status()

    def _manage_exclusions(self):
        """Open the exclusions management dialog."""
        if self.index is None:
            messagebox.showwarning("No Index", "Load or create an index first.")
            return

        ExclusionsDialog(self, self.index, self._on_exclusions_changed)

    def _group_by_subject(self):
        """Group images in the index by subject."""
        if self.index is None:
            messagebox.showwarning("No Index", "Load or create an index first.")
            return

        if self.index.count < 2:
            messagebox.showwarning("Not Enough Images", "Need at least 2 images to group.")
            return

        progress = ProgressDialog(self, "Grouping Images...")
        progress.set_status("Clustering images by similarity...")

        def do_group():
            try:
                groups = group_by_subject(self.index)
                self._safe_after(lambda: self._on_grouping_complete(groups, progress))
            except Exception as e:
                self._safe_after(lambda: self._on_error(str(e), progress))

        thread = threading.Thread(target=do_group)
        thread.start()

    def _on_grouping_complete(self, groups, progress: ProgressDialog):
        """Handle grouping completion."""
        if self._shutting_down:
            return
        if not progress._destroyed:
            progress.destroy()

        if not groups:
            messagebox.showinfo("No Groups", "Could not form any groups.")
        else:
            def on_rename(updated_groups):
                renames = rename_images_by_group(updated_groups, dry_run=False)
                # Refresh index if files were renamed
                if renames and self.index.source_directory:
                    self.index = ImageIndex.create_from_directory(self.index.source_directory)
                    self._update_status()

            GroupingDialog(self, groups, on_rename)
