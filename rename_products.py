"""
Renames all files in data/products/<category>/ to short, safe filenames
(e.g. shoes_0.jpg, shoes_1.jpg, ...). Run this once from the project root:

    python rename_products.py
"""
import os

for cat in ["shoes", "electronics", "clothing"]:
    folder = f"data/products/{cat}"
    if not os.path.exists(folder):
        print(f"Skipping missing folder: {folder}")
        continue
    files = sorted(os.listdir(folder))
    for i, fname in enumerate(files):
        old_path = os.path.join(folder, fname)
        if not os.path.isfile(old_path):
            continue
        ext = os.path.splitext(fname)[1].lower() or ".jpg"
        if ext not in [".jpg", ".jpeg", ".png"]:
            ext = ".jpg"
        new_name = f"{cat}_{i}{ext}"
        new_path = os.path.join(folder, new_name)
        if old_path != new_path:
            os.rename(old_path, new_path)
    print(f"{cat}: renamed {len(files)} files")

print("Done. Now re-run: python app/services/product_classifier.py")
