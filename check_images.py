import os
import torch
import clip
from PIL import Image
from tqdm import tqdm

# Setup
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Config
PHOTO_LIBRARY_PATH = "C:\\Users\\yinju\\Pictures\\yuki"
SIMILARITY_THRESHOLD = 0.997  # tweak between 0.90–0.97 as needed

def get_image_embedding(image_path: str) -> torch.Tensor:
    try:
        image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = model.encode_image(image)
        return emb / emb.norm(dim=-1, keepdim=True)
    except Exception as e:
        print(f"Error with image {image_path}: {e}")
        return None

def index_library_embeddings(library_path: str):
    embeddings = []
    print("Indexing photo library...")
    for fname in tqdm(os.listdir(library_path)):
        full_path = os.path.join(library_path, fname)
        if os.path.isfile(full_path):
            emb = get_image_embedding(full_path)
            if emb is not None:
                embeddings.append((fname, emb))
    return embeddings

def find_duplicates_within_library(library_embeddings):
    print("\nChecking for duplicates within the photo library...")
    duplicates = []
    checked = set()
    for i, (fname1, emb1) in enumerate(tqdm(library_embeddings)):
        for j, (fname2, emb2) in enumerate(library_embeddings):
            if i >= j:
                continue  # Avoid self-comparison and duplicate pairs
            pair = tuple(sorted([fname1, fname2]))
            if pair in checked:
                continue
            similarity = (emb1 @ emb2.T).item()
            if similarity >= SIMILARITY_THRESHOLD:
                duplicates.append((fname1, fname2, similarity))
            checked.add(pair)
    return duplicates

if __name__ == "__main__":
    library_embeddings = index_library_embeddings(PHOTO_LIBRARY_PATH)
    duplicates = find_duplicates_within_library(library_embeddings)

    print("\nSummary of duplicates:")
    if duplicates:
        for fname1, fname2, sim in duplicates:
            print(f"[DUPLICATE] {fname1} ≈ {fname2} (similarity={sim:.4f})")
    else:
        print("No duplicates found within the photo library.")