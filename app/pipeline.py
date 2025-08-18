from PIL import Image
import numpy as np
import torch
import os
from .models.models import load_segformer, load_classifier
from .utils.helpers import segment_large_image, extract_blobs, compute_patch_dominance
from torchvision import transforms
import pandas as pd
import cv2
import io

CLASS_NAMES = {0: "background", 1: "spartina", 2: "puccinellia"}
PATCH_SIZE_PX = 800

seg_model, seg_processor, DEVICE = load_segformer()
classifier = load_classifier(device=DEVICE)

def run_inference(image_input, image_name="input_image"):
    # Load image from path or bytes
    if isinstance(image_input, bytes):
        img_pil = Image.open(io.BytesIO(image_input)).convert("RGB")
    elif isinstance(image_input, str):
        img_pil = Image.open(image_input).convert("RGB")
    else:
        raise ValueError("Unsupported input format")

    seg_mask = segment_large_image(img_pil, seg_model, seg_processor, DEVICE)

    # === BLOB CLASSIFICATION ===
    boxes = extract_blobs(seg_mask.copy())
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    for (x, y, w, h) in boxes:
        blob_crop = img_pil.crop((x, y, x + w, y + h))
        input_tensor = transform(blob_crop).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            pred = classifier(input_tensor).argmax(1).item()
        seg_mask[y:y+h, x:x+w] = pred + 1

    # === DOMINANCE SCORING ===
    dominance = compute_patch_dominance(seg_mask, PATCH_SIZE_PX)
    for d in dominance:
        d["image"] = image_name

    spartina_count = sum(1 for row in dominance if row["dominant"] == "spartina")
    puccinellia_count = sum(1 for row in dominance if row["dominant"] == "puccinellia")

    if spartina_count == puccinellia_count == 0:
        dominant = "none"
    elif spartina_count >= puccinellia_count:
        dominant = "spartina"
    else:
        dominant = "puccinellia"

    return {
        "image": image_name,
        "dominance": dominance,
        "summary": {
            "spartina_count": spartina_count,
            "puccinellia_count": puccinellia_count,
            "dominant_species": dominant
        }
    }
