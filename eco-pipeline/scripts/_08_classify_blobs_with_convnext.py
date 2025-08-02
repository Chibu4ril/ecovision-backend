import os
import torch
import torchvision.transforms as transforms
from torchvision.models import convnext_base, ConvNeXt_Base_Weights
from PIL import Image
from tqdm import tqdm
from torch import nn

# === Paths ===
BLOB_DIR = os.path.join("outcome_folders", "_06_extracted_blobs_with_coords")
MODEL_PATH = os.path.join("trained_models", "convnext_classifier_base_with_coords.pth")
OUTPUT_DIR = os.path.join("outcome_folders", "_07_classified_blobs_with_coords")

# === Config ===
IMAGE_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Load Model ===
model = convnext_base(weights=ConvNeXt_Base_Weights.IMAGENET1K_V1)
model.classifier[2] = nn.Linear(model.classifier[2].in_features, 2)

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
CLASS_NAMES = checkpoint["class_names"]

model.to(DEVICE)
model.eval()

# === Transform ===
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], 
                         [0.229, 0.224, 0.225])
])

# === Collect all blob image files ===
blob_files = []
for root, _, files in os.walk(BLOB_DIR):
    for f in files:
        if f.lower().endswith(".png"):
            blob_files.append(os.path.join(root, f))

# === Ensure output directory exists ===
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Inference + Save with Coordinates ===
for img_path in tqdm(blob_files, desc="Classifying blobs with coords"):
    image = Image.open(img_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)

    # Get matching coordinate .txt file
    txt_path = img_path.replace(".png", ".txt")
    if not os.path.exists(txt_path):
        continue

    with open(txt_path, "r") as f:
        contents = f.read().strip().split()
        if len(contents) < 3:
            continue
        label = contents[0]
        coords = list(map(int, contents[1:])) if len(contents) == 5 else [0, 0, 0, 0]
        x, y, w, h = coords if len(coords) == 4 else (0, 0, 0, 0)

    # Predict label
    with torch.no_grad():
        output = model(input_tensor)
        pred_idx = output.argmax(dim=1).item()
        pred_label = CLASS_NAMES[pred_idx]

    # Save outputs
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    img_save_path = os.path.join(OUTPUT_DIR, base_name + ".png")
    label_save_path = os.path.join(OUTPUT_DIR, base_name + ".txt")

    image.save(img_save_path)
    with open(label_save_path, "w") as f:
        f.write(f"{pred_label} {x} {y} {w} {h}")

print(f"✅ Done. {len(blob_files)} blobs classified with coords and saved to: {OUTPUT_DIR}")
