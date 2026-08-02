"""
Renames all files in data/products/<category>/ to short, safe filenames
(e.g. shoes_0.jpg, shoes_1.jpg, ...). Run this once from the project root:

    python rename_products.py
"""
import os
import uuid

for cat in ["shoes", "electronics", "clothing"]:
    folder = f"data/products/{cat}"
    if not os.path.exists(folder):
        print(f"Skipping missing folder: {folder}")
        continue
    files = [f for f in sorted(os.listdir(folder)) if os.path.isfile(os.path.join(folder, f))]

    # Phase 1: rename everything to unique temp names to avoid collisions
    temp_names = []
    for fname in files:
        old_path = os.path.join(folder, fname)
        ext = os.path.splitext(fname)[1].lower()
        if ext not in [".jpg", ".jpeg", ".png"]:
            ext = ".jpg"
        temp_name = f"tmp_{uuid.uuid4().hex}{ext}"
        os.rename(old_path, os.path.join(folder, temp_name))
        temp_names.append(temp_name)

    # Phase 2: rename temp names to final clean names
    for i, temp_name in enumerate(temp_names):
        ext = os.path.splitext(temp_name)[1]
        final_name = f"{cat}_{i}{ext}"
        os.rename(os.path.join(folder, temp_name), os.path.join(folder, final_name))

    print(f"{cat}: renamed {len(temp_names)} files")

print("Done. Now re-run: python app/services/product_classifier.py")