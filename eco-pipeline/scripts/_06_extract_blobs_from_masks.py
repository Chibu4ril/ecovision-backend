import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

# === CONFIG ===
PRED_MASKS_DIR = "outcome_folders\_05_predicted_masks\\test"
ORIG_IMAGES_DIR = "outcome_folders\_04_final_split_dataset\\test\images"
OUTPUT_DIR = "outcome_folders\_06_extracted_blobs_with_coords"
CLASS_NAMES = {1: "spartina", 2: "puccinellia"}
MIN_BLOB_AREA = 100

os.makedirs(OUTPUT_DIR, exist_ok=True)

for label_name in CLASS_NAMES.values():
    os.makedirs(os.path.join(OUTPUT_DIR, label_name), exist_ok=True)

# === Helper ===
def extract_blobs(mask, label_value):
    mask_bin = np.uint8(mask == label_value) * 255
    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [cnt for cnt in contours if cv2.contourArea(cnt) > MIN_BLOB_AREA]

# === Process All Masks ===
for filename in tqdm(sorted(os.listdir(PRED_MASKS_DIR)), desc="Extracting blobs"):
    if not filename.endswith(".png"):
        continue

    mask_path = os.path.join(PRED_MASKS_DIR, filename)
    image_path = os.path.join(ORIG_IMAGES_DIR, filename)

    mask = np.array(Image.open(mask_path))
    image = np.array(Image.open(image_path).convert("RGB"))

    for class_id, class_name in CLASS_NAMES.items():
        blobs = extract_blobs(mask, class_id)
        for i, contour in enumerate(blobs):
            x, y, w, h = cv2.boundingRect(contour)
            crop = image[y:y+h, x:x+w]

            base_name = f"{filename[:-4]}_{class_name}_{i}"
            img_path = os.path.join(OUTPUT_DIR, class_name, base_name + ".png")
            txt_path = os.path.join(OUTPUT_DIR, class_name, base_name + ".txt")

            cv2.imwrite(img_path, crop)

            with open(txt_path, "w") as f:
                f.write(f"{class_name} {x} {y} {w} {h}")

print("✅ Done. Blobs with coordinates saved to:", OUTPUT_DIR)
