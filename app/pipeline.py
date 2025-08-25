from PIL import Image
import torch
import os
from .models.models import get_models, load_classifier
from .utils.helpers import segment_large_image, extract_blobs, compute_patch_dominance, visualize_grid
from torchvision import transforms
import pandas as pd
import cv2
import io
from config.config import supabase
from .utils.gradcam import gradcam_full_image
import uuid
import tempfile
from fastapi import HTTPException 



CLASS_NAMES = {0: "background", 1: "spartina", 2: "puccinellia"}
PATCH_SIZE_PX = 800



# seg_model, seg_processor, DEVICE = get_models()
# classifier = load_classifier(device=DEVICE)
# last_conv_layer = classifier.features[-1]


def run_inference(image_input, image_name="input_image", user_id=None, supabase=None):
    seg_model, seg_processor, DEVICE, classifier = get_models()
    
    if not user_id:
        raise HTTPException(status_code=401, detail="Missing user_id")
    if supabase is None:
        raise HTTPException(status_code=500, detail="Supabase client not provided")
    

    if not user_id:
        raise HTTPException(status_code=401, detail="Missing user_id")
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

    gradcam_img = img_pil.copy()  # keep a copy for Grad-CAM overlays
    gradcam_url = None

    # Collect predictions per blob
    blob_results = []
    for (x, y, w, h) in boxes:
        blob_crop = img_pil.crop((x, y, x + w, y + h))
        input_tensor = transform(blob_crop).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            pred = classifier(input_tensor).argmax(1).item()
            seg_mask[y:y+h, x:x+w] = pred + 1
            blob_results.append(((x, y, w, h), blob_crop, pred))


    # === GRAD-CAM overlays ===
    gradcam_img = img_pil.copy()
    for (x, y, w, h), blob_crop, pred in blob_results:
        input_tensor = transform(blob_crop).unsqueeze(0).to(DEVICE)
        gradcam_img = gradcam_full_image(
            classifier, input_tensor, pred, last_conv_layer, gradcam_img, (x, y, w, h)
        )

    # Upload once after all blobs are applied
    gradcam_bucket = "gradcam_overlays"
    gradcam_file_path = f"{image_name}_gradcam_{uuid.uuid4().hex}.png"

    gradcam_bytes = io.BytesIO()
    gradcam_img.save(gradcam_bytes, format="PNG")
    gradcam_bytes.seek(0)

    # Convert to raw bytes
    gradcam_content = gradcam_bytes.getvalue()
    res = None
    try:
        # Upload raw bytes
        response = supabase.storage.from_(gradcam_bucket).upload(gradcam_file_path,gradcam_content,{"content-type": "image/png"},)

        if getattr(response, "error", None):
            raise Exception(f"Grad-CAM upload failed: {response.error}")

        gradcam_url = supabase.storage.from_(gradcam_bucket).get_public_url(gradcam_file_path)


    except Exception as e:
        raise Exception(f"Grad-CAM upload failed: {e}")



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

  
    # === OVERLAY VISUALIZATION ===
    overlay_img = visualize_grid(img_pil, dominance, PATCH_SIZE_PX, save_path=None)

    # Convert overlay to bytes
    img_bytes = io.BytesIO()
    overlay_img.save(img_bytes, format="PNG")
    img_bytes.seek(0)
    overlay_content = img_bytes.getvalue()

    bucket_name = "overlays"
    # file_path = f"{image_name}_overlay.png"
    file_path = f"{image_name}_overlay_{uuid.uuid4().hex}.png"

    try:
        res = supabase.storage.from_(bucket_name).upload(
            path=file_path,
            file=overlay_content,
            file_options={"content-type": "image/png"}
        )
    except Exception as e:
        raise Exception(f"Overlay upload failed: {str(e)}")

    overlay_url = supabase.storage.from_(bucket_name).get_public_url(file_path)

    if not overlay_url or not isinstance(overlay_url, str):
        raise Exception("Overlay upload failed: could not retrieve public URL")
    
    
    supabase.table("processed_images").insert({
        "original_filename": image_name,
        "overlay_path": file_path,
        "overlay_url": overlay_url,
        "gradcam_path": gradcam_file_path,
        "gradcam_url": gradcam_url,
        "spartina_count": spartina_count,
        "puccinellia_count": puccinellia_count,
        "dominant_species": dominant,
        "user_id": user_id 
    }).execute()

    # return {
    #     "image": image_name,
    #     "overlay_url": overlay_url,
    #     "gradcam_url": gradcam_url,
    #     "dominance": dominance,
    #     "summary": {
    #         "spartina_count": spartina_count,
    #         "puccinellia_count": puccinellia_count,
    #         "dominant_species": dominant
    #     }
    # }

    return {
    "status": "success",
    "image": image_name,
    }


 