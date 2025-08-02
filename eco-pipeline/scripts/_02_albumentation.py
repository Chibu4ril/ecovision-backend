import os
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm
import albumentations as A

# === Input/output paths ===
input_images = sorted(glob("outcome_folders/_02_dataset_json_to_img_ready/images/*.png"))
input_masks = sorted(glob("outcome_folders/_02_dataset_json_to_img_ready/masks/*.png"))

out_img_dir = "outcome_folders/_03_augmented_dataset/images"
out_mask_dir = "outcome_folders/_03_augmented_dataset/masks"

os.makedirs(out_img_dir, exist_ok=True)
os.makedirs(out_mask_dir, exist_ok=True)

# === Albumentation pipeline ===
transform = A.Compose([
    A.Resize(512, 512),
    A.HorizontalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, border_mode=cv2.BORDER_REFLECT, rotate_limit=20, p=0.5),
], additional_targets={'mask': 'mask'})

# === Number of augmentations per image ===
AUGS_PER_IMAGE = 5

print(f"🔄 Generating {AUGS_PER_IMAGE} augmentations per image...")

img_count = 0
for i, (img_path, mask_path) in enumerate(tqdm(zip(input_images, input_masks), total=len(input_images))):
    img = cv2.imread(img_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if img is None or mask is None:
        print(f"⚠️ Skipped {img_path} or {mask_path}")
        continue

    if img.shape[:2] != mask.shape[:2]:
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

    for j in range(AUGS_PER_IMAGE):
        try:
            augmented = transform(image=img, mask=mask)
            aug_img = augmented["image"]
            aug_mask = augmented["mask"]

            base_name = f"aug_{img_count:04d}.png"
            cv2.imwrite(os.path.join(out_img_dir, base_name), aug_img)
            cv2.imwrite(os.path.join(out_mask_dir, base_name), aug_mask)

            img_count += 1
        except Exception as e:
            print(f"❌ Error: {e}")
            continue

print(f"✅ Created {img_count} augmented image-mask pairs in `augmented_dataset/`")
