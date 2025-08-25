import numpy as np
import cv2
from PIL import Image
import torch

def segment_large_image(img_pil, model, processor, device, tile_size=640, overlap=64):
    img_np = np.array(img_pil)
    h, w, _ = img_np.shape
    full_mask = np.zeros((h, w), dtype=np.uint8)

    for y in range(0, h, tile_size - overlap):
        for x in range(0, w, tile_size - overlap):
            x_end = min(x + tile_size, w)
            y_end = min(y + tile_size, h)

            tile = img_pil.crop((x, y, x_end, y_end))
            inputs = processor(images=tile, return_tensors="pt").to(device)

            with torch.no_grad():
                output = model(**inputs)
                pred = output.logits.argmax(dim=1).squeeze().cpu().numpy()

            pred_resized = cv2.resize(pred.astype(np.uint8), (x_end - x, y_end - y), interpolation=cv2.INTER_NEAREST)
            full_mask[y:y_end, x:x_end] = pred_resized

    return full_mask

def extract_blobs(mask):
    blob_mask = (mask == 1) | (mask == 2)
    contours, _ = cv2.findContours(blob_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [cv2.boundingRect(cnt) for cnt in contours if cv2.contourArea(cnt) > 50]

def compute_patch_dominance(mask, patch_size):
    h, w = mask.shape
    results = []
    for y in range(0, h, patch_size):
        for x in range(0, w, patch_size):
            patch = mask[y:y+patch_size, x:x+patch_size]
            total = patch.size
            counts = [(patch == i).sum() for i in range(3)]
            percents = [round(100 * c / total, 2) for c in counts]
            dominant_idx = np.argmax(percents[1:]) + 1 if sum(percents[1:]) > 0 else 0
            results.append({
                "x": x // patch_size,
                "y": y // patch_size,
                "dominant": {0: "background", 1: "spartina", 2: "puccinellia"}[dominant_idx],
                "percent": percents[dominant_idx],
                "total_blobs": sum(counts[1:])
            })
    return results

def visualize_grid(pil_img, dominance_data, patch_size, save_path=None):
    img = np.array(pil_img)
    vis = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    for d in dominance_data:
        x = d["x"] * patch_size
        y = d["y"] * patch_size
        label = d["dominant"]
        perc = d["percent"]
        color = (0, 255, 0) if label == "spartina" else (255, 0, 0)
        cv2.rectangle(vis, (x, y), (x + patch_size, y + patch_size), color, 2)
        cv2.putText(vis, f"{label}: {perc}%", (x + 5, y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    if save_path:
        cv2.imwrite(save_path, vis)
        return None
    else:
        vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
        return Image.fromarray(vis_rgb)