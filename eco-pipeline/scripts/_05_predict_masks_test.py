import os
import torch
from PIL import Image
import numpy as np
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
from tqdm import tqdm

# === CONFIG ===
MODEL_DIR = "trained_models\segformer"
TEST_IMAGES_DIR = "outcome_folders\_04_final_split_dataset\\test\images"
OUTPUT_MASKS_DIR = "outcome_folders\_05_predicted_masks\\test"
IMAGE_SIZE = 512
NUM_CLASSES = 3

# === Setup ===
os.makedirs(OUTPUT_MASKS_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# === Load Model & Preprocessor ===
model = SegformerForSemanticSegmentation.from_pretrained(MODEL_DIR).to(device)
processor = SegformerImageProcessor(do_resize=True, size=IMAGE_SIZE, do_normalize=True)

# === Predict Function ===
def predict_mask(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=np.array(image), return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        pred_mask = torch.argmax(logits, dim=1).squeeze().cpu().numpy().astype(np.uint8)
    return pred_mask

# === Predict All Test Images ===
image_files = sorted([f for f in os.listdir(TEST_IMAGES_DIR) if f.endswith(".png")])

for fname in tqdm(image_files, desc="Predicting masks"):
    input_path = os.path.join(TEST_IMAGES_DIR, fname)
    pred_mask = predict_mask(input_path)
    save_path = os.path.join(OUTPUT_MASKS_DIR, fname)
    Image.fromarray(pred_mask).save(save_path)

print("✅ Done. All predicted masks saved to:", OUTPUT_MASKS_DIR)
