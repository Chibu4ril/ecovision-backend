import os
import shutil
import random
from tqdm import tqdm

# Set paths
input_images_dir = "outcome_folders/_03_augmented_dataset/images" 
input_masks_dir = "outcome_folders/_03_augmented_dataset/masks"
output_base = "outcome_folders/_04_final_split_dataset"

# Define split ratios
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

# Create output folders
splits = ["train", "val", "test"]
for split in splits:
    os.makedirs(os.path.join(output_base, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_base, split, "masks"), exist_ok=True)

# Get list of image files
image_filenames = sorted(os.listdir(input_images_dir))
random.shuffle(image_filenames)  # Shuffle to ensure randomness

# Split indices
total = len(image_filenames)
train_end = int(train_ratio * total)
val_end = train_end + int(val_ratio * total)

train_files = image_filenames[:train_end]
val_files = image_filenames[train_end:val_end]
test_files = image_filenames[val_end:]

# Helper to copy images and masks
def copy_files(file_list, split_name):
    for filename in tqdm(file_list, desc=f"Copying {split_name}"):
        img_src = os.path.join(input_images_dir, filename)
        mask_src = os.path.join(input_masks_dir, filename)

        img_dst = os.path.join(output_base, split_name, "images", filename)
        mask_dst = os.path.join(output_base, split_name, "masks", filename)

        if os.path.exists(img_src) and os.path.exists(mask_src):
            shutil.copy(img_src, img_dst)
            shutil.copy(mask_src, mask_dst)
        else:
            print(f"❌ Missing file for {filename}")

# Perform copy
copy_files(train_files, "train")
copy_files(val_files, "val")
copy_files(test_files, "test")

print("✅ Dataset split into train/val/test and saved to `final_dataset/`")
