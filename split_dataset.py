import os
import shutil
import random

# Set fixed seed for reproducibility (mandatory for the assignment)
random.seed(42)

# Source directories
IMAGE_DIR = "C:/Users/admin/Downloads/archive/data/images"
LABEL_DIR = "C:/Users/admin/Downloads/archive/data/labels-YOLO"

# Output destination directory
OUTPUT_DIR = "C:/Users/admin/Downloads/dataset"

# Target split ratios (70% train, 15% val, 15% test)
SPLITS = {
    "train": 0.70,
    "val": 0.15,
    "test": 0.15
}

# Create required folder structure
for split in SPLITS.keys():
    os.makedirs(os.path.join(OUTPUT_DIR, "images", split), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "labels", split), exist_ok=True)

# Collect all valid image-label pairs
valid_extensions = (".jpg", ".jpeg", ".png")
all_images = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(valid_extensions)]

matched_pairs = []
for img_name in all_images:
    base_name = os.path.splitext(img_name)[0]
    label_name = f"{base_name}.txt"
    label_path = os.path.join(LABEL_DIR, label_name)
    
    # Ensure label file exists before pairing
    if os.path.exists(label_path):
        matched_pairs.append((img_name, label_name))
    else:
        print(f"Warning: Missing label for {img_name}, skipping.")

print(f"Total matched pairs: {len(matched_pairs)}")

# Shuffle deterministically
random.shuffle(matched_pairs)

# Compute split counts
total = len(matched_pairs)
n_train = int(total * SPLITS["train"])
n_val = int(total * SPLITS["val"])

train_pairs = matched_pairs[:n_train]
val_pairs = matched_pairs[n_train:n_train + n_val]
test_pairs = matched_pairs[n_train + n_val:]

split_data = {
    "train": train_pairs,
    "val": val_pairs,
    "test": test_pairs
}

# Copy files to their split directories
for split_name, pairs in split_data.items():
    print(f"Copying {len(pairs)} files to {split_name} split...")
    for img_name, lbl_name in pairs:
        # Copy image
        shutil.copy(
            os.path.join(IMAGE_DIR, img_name),
            os.path.join(OUTPUT_DIR, "images", split_name, img_name)
        )
        # Copy label
        shutil.copy(
            os.path.join(LABEL_DIR, lbl_name),
            os.path.join(OUTPUT_DIR, "labels", split_name, lbl_name)
        )

print("Dataset preparation and split completed successfully!")