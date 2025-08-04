import os
import cv2
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms, models
from torchvision.models import ConvNeXt_Base_Weights
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor, SegformerConfig
from safetensors.torch import load_file
from tqdm import tqdm


logging.basicConfig(filename="/tmp/script_output.log", level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

# print(sys.argv)
if len(sys.argv) < 2:
    print("Error: Missing required arguments. Usage: script.py <file_url>")
    sys.exit(1)


file_url = sys.argv[1] 


# === CONFIG ===
UAV_IMAGE_DIR = "eco-pipeline/uav_images"
os.makedirs(UAV_IMAGE_DIR, exist_ok=True)

try:
    response = requests.get(file_url, stream=True)
    response.raise_for_status()

    filename = os.path.basename(file_url.split("?")[0])  # strip any query params
    local_path = os.path.join(UAV_IMAGE_DIR, filename)

    with open(local_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    logging.info(f"Downloaded image to: {local_path}")
    print(f"Downloaded image to: {local_path}")

except requests.exceptions.RequestException as e:
    logging.error(f"Failed to download image: {e}")
    print(f"Error: Failed to download image: {e}")
    sys.exit(1)


SEGFORMER_CONFIG_PATH = "eco-pipeline/trained_models/segformer/config.json"
SEGFORMER_WEIGHTS_PATH = "eco-pipeline/trained_models/segformer/model.safetensors"
CLASSIFIER_MODEL_PATH = "eco-pipeline/trained_models/convnext_classifier_base_with_coords.pth"
OUTPUT_DIR = "eco-pipeline/outputs"
PIXELS_PER_METER = 200
PATCH_SIZE_METERS = 4
PATCH_SIZE_PX = PIXELS_PER_METER * PATCH_SIZE_METERS

CLASS_NAMES = {0: "background", 1: "spartina", 2: "puccinellia"}
NUM_CLASSES = len(CLASS_NAMES)
NUM_CLASSES_CONV = 2  

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === LOAD SEGFORMER FROM LOCAL FILES ===
config = SegformerConfig.from_json_file(SEGFORMER_CONFIG_PATH)
seg_model = SegformerForSemanticSegmentation(config)
seg_model.load_state_dict(load_file(SEGFORMER_WEIGHTS_PATH))
seg_model.to(DEVICE).eval()
seg_processor = SegformerImageProcessor(do_resize=True, size=640, do_normalize=True)

# === LOAD CONVNEXT CLASSIFIER ===
weights = ConvNeXt_Base_Weights.DEFAULT
classifier = models.convnext_base(weights=weights)
classifier.classifier[2] = torch.nn.Linear(classifier.classifier[2].in_features, NUM_CLASSES_CONV)
classifier.load_state_dict(torch.load(CLASSIFIER_MODEL_PATH, map_location=DEVICE)["model_state_dict"])
classifier.eval().to(DEVICE)

# === TRANSFORMS ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# === FUNCTION: Extract Blobs ===
def extract_blobs(mask):
    blob_mask = (mask == 1) | (mask == 2)  # exclude background
    contours, _ = cv2.findContours(blob_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [cv2.boundingRect(cnt) for cnt in contours if cv2.contourArea(cnt) > 50]

# === FUNCTION: Compute Patch Dominance ===
def compute_patch_dominance(mask, patch_size):
    h, w = mask.shape
    results = []

    for y in range(0, h, patch_size):
        for x in range(0, w, patch_size):
            patch = mask[y:y+patch_size, x:x+patch_size]
            total = patch.size
            counts = [(patch == i).sum() for i in range(NUM_CLASSES)]
            percents = [round(100 * c / total, 2) for c in counts]
            dominant_idx = np.argmax(percents[1:]) + 1 if sum(percents[1:]) > 0 else 0
            results.append({
                "image": "",
                "x": x // patch_size,
                "y": y // patch_size,
                "dominant": CLASS_NAMES[dominant_idx],
                "percent": percents[dominant_idx],
                "total_blobs": sum(counts[1:])
            })
    return results

# === FUNCTION: Visual Overlay ===
def visualize_grid(pil_img, dominance_data, patch_size, save_path):
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
    cv2.imwrite(save_path, vis)


# === FUNCTION: Patch-wise Segmentation for Large UAV Images ===
def segment_large_image(img_pil, tile_size=640, overlap=64):
    """
    Splits large image into tiles, performs segmentation, and reconstructs full mask.
    """
    img_np = np.array(img_pil)
    h, w, _ = img_np.shape
    full_mask = np.zeros((h, w), dtype=np.uint8)

    for y in range(0, h, tile_size - overlap):
        for x in range(0, w, tile_size - overlap):
            x_end = min(x + tile_size, w)
            y_end = min(y + tile_size, h)

            tile = img_pil.crop((x, y, x_end, y_end))
            inputs = seg_processor(images=tile, return_tensors="pt").to(DEVICE)

            with torch.no_grad():
                output = seg_model(**inputs)
                pred = output.logits.argmax(dim=1).squeeze().cpu().numpy()

            # Resize prediction back to tile's actual size (in case of model resizing)
            pred_resized = cv2.resize(pred.astype(np.uint8), (x_end - x, y_end - y), interpolation=cv2.INTER_NEAREST)

            full_mask[y:y_end, x:x_end] = pred_resized

    return full_mask


# === MAIN LOOP ===
for filename in tqdm(os.listdir(UAV_IMAGE_DIR), desc="Processing UAV Images"):
    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    name = os.path.splitext(filename)[0]
    img_path = os.path.join(UAV_IMAGE_DIR, filename)
    img_pil = Image.open(img_path).convert("RGB")

    # === SEGMENTATION ===
    seg_mask = segment_large_image(img_pil, tile_size=640, overlap=64)

    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{name}_segmask.png"), seg_mask.astype(np.uint8))

    # === BLOB CLASSIFICATION ===
    boxes = extract_blobs(seg_mask.copy())
    for (x, y, w, h) in boxes:
        blob_crop = img_pil.crop((x, y, x + w, y + h))
        input_tensor = transform(blob_crop).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            pred = classifier(input_tensor).argmax(1).item()
        seg_mask[y:y+h, x:x+w] = pred + 1  # 0→1 spartina, 1→2 puccinellia

    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{name}_refined.png"), seg_mask.astype(np.uint8))

   # === DOMINANCE SCORING + SUMMARY GENERATION ===
summary_records = []

for filename in tqdm(os.listdir(UAV_IMAGE_DIR), desc="Processing UAV Images"):
    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    name = os.path.splitext(filename)[0]
    img_path = os.path.join(UAV_IMAGE_DIR, filename)
    img_pil = Image.open(img_path).convert("RGB")

    # === SEGMENTATION ===
    seg_mask = segment_large_image(img_pil, tile_size=640, overlap=64)
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{name}_segmask.png"), seg_mask.astype(np.uint8))

    # === BLOB CLASSIFICATION ===
    boxes = extract_blobs(seg_mask.copy())
    for (x, y, w, h) in boxes:
        blob_crop = img_pil.crop((x, y, x + w, y + h))
        input_tensor = transform(blob_crop).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            pred = classifier(input_tensor).argmax(1).item()
        seg_mask[y:y+h, x:x+w] = pred + 1  # 0→1 spartina, 1→2 puccinellia

    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{name}_refined.png"), seg_mask.astype(np.uint8))

    # === DOMINANCE SCORING ===
    dominance = compute_patch_dominance(seg_mask, PATCH_SIZE_PX)
    for d in dominance:
        d["image"] = name
    df = pd.DataFrame(dominance)
    csv_path = os.path.join(OUTPUT_DIR, f"{name}_dominance.csv")
    df.to_csv(csv_path, index=False)

    # === OVERLAY VISUALIZATION ===
    vis_path = os.path.join(OUTPUT_DIR, f"{name}_overlay.png")
    visualize_grid(img_pil, dominance, PATCH_SIZE_PX, vis_path)

    # === SUMMARY FROM CSV ===
    spartina_count = sum(1 for row in dominance if row["dominant"] == "spartina")
    puccinellia_count = sum(1 for row in dominance if row["dominant"] == "puccinellia")

    if spartina_count == puccinellia_count == 0:
        dominant = "none"
    elif spartina_count >= puccinellia_count:
        dominant = "spartina"
    else:
        dominant = "puccinellia"

    summary_records.append({
        "image": name,
        "spartina_count": spartina_count,
        "puccinellia_count": puccinellia_count,
        "dominant_species": dominant
    })

# === SAVE FINAL SUMMARY ===
summary_df = pd.DataFrame(summary_records)
summary_csv_path = os.path.join(OUTPUT_DIR, "dominance_summary.csv")
summary_df.to_csv(summary_csv_path, index=False)

print(f"✅ All UAV images processed. Dominance summary saved to: {summary_csv_path}")
