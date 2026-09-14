import os
import shutil
import tensorflow as tf

# Paths identified directly from your explorer
source_pet_images = os.path.join("datasets", "cats_and_dogs_extracted", "PetImages")
target_dir = "dataset"

# Fallback check if PetImages is directly inside or nested
if not os.path.exists(source_pet_images):
    for root, dirs, files in os.walk("datasets"):
        if "Cat" in dirs and "Dog" in dirs:
            source_pet_images = root
            break

print(f"Source folder found: {source_pet_images}")

os.makedirs(target_dir, exist_ok=True)

# Move Cat and Dog folders to ./dataset
for animal in ["Cat", "Dog"]:
    src = os.path.join(source_pet_images, animal)
    dst = os.path.join(target_dir, animal)
    if not os.path.exists(dst) and os.path.exists(src):
        print(f"Moving {animal} directory...")
        shutil.move(src, dst)

# Filter out corrupted images
print("\nFiltering corrupted images...")
num_skipped = 0
for folder_name in ("Cat", "Dog"):
    folder_path = os.path.join(target_dir, folder_name)
    if not os.path.exists(folder_path):
        continue
    for fname in os.listdir(folder_path):
        fpath = os.path.join(folder_path, fname)
        try:
            with open(fpath, "rb") as f:
                is_jfif = tf.compat.as_bytes("JFIF") in f.peek(10)
            if not is_jfif:
                num_skipped += 1
                os.remove(fpath)
        except Exception:
            num_skipped += 1
            if os.path.exists(fpath):
                os.remove(fpath)

print(f"Done! Cleaned up {num_skipped} unreadable files.")
print("Dataset ready in ./dataset/Cat and ./dataset/Dog!")