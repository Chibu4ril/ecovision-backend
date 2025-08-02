import os
import shutil
import glob
import json
from labelme import utils
import numpy as np
import PIL.Image

# === Paths ===
annotated_dir = "outcome_folders\_01_annotated_dataset"
image_dir = "dataset"
output_dir = "outcome_folders\_02_dataset_json_to_img_ready"
images_out = os.path.join(output_dir, "images")
masks_out = os.path.join(output_dir, "masks")
os.makedirs(images_out, exist_ok=True)
os.makedirs(masks_out, exist_ok=True)

# === Fixed label values ===
label_name_to_value = {
    "_background_": 0,
    "Spartina_maritima": 1,
    "Puccinellia_maritima": 2
}

# Optional mapping for auto-renaming incorrect JSON labels
label_correction = {
    "spartina": "Spartina_maritima",
    "puccinellia": "Puccinellia_maritima"
}

species_dirs = {
    "Spartina_maritima": {
        "json_dir": os.path.join(annotated_dir, "spartina"),
        "img_dir": os.path.join(image_dir, "Spartina_maritima"),
        "label_value": 1
    },
    "Puccinellia_maritima": {
        "json_dir": os.path.join(annotated_dir, "puccinellia"),
        "img_dir": os.path.join(image_dir, "Puccinellia_maritima"),
        "label_value": 2
    }
}

counter = {
    "Spartina_maritima": 1,
    "Puccinellia_maritima": 1
}

print("🚀 Starting JSON-to-mask processing...")

for species_name, paths in species_dirs.items():
    json_files = sorted(glob.glob(os.path.join(paths["json_dir"], "*.json")))
    print(f"🧠 Processing {species_name}: {len(json_files)} annotations")

    for json_file in json_files:
        base_name = os.path.splitext(os.path.basename(json_file))[0]

        # Original image path
        image_path = os.path.join(paths["img_dir"], f"{base_name}.png")
        if not os.path.exists(image_path):
            print(f"⚠️ Skipping {base_name}: image not found.")
            continue

        # New name
        count = counter[species_name]
        new_name = f"{species_name.replace(' ', '_')}_{count:03d}"
        counter[species_name] += 1

        # Load image
        with open(json_file) as f:
            data = json.load(f)

        # 🔁 Normalize labels to match expected values
        for shape in data["shapes"]:
            original_label = shape["label"]
            if original_label in label_correction:
                shape["label"] = label_correction[original_label]

        img = utils.img_b64_to_arr(data["imageData"]) if data.get("imageData") else np.asarray(PIL.Image.open(image_path))

        try:
            lbl, _ = utils.shapes_to_label(img.shape, data["shapes"], label_name_to_value)
        except KeyError as e:
            print(f"❌ Label not found in map: {e} in {json_file}")
            continue

        # Save output
        mask_out_path = os.path.join(masks_out, f"{new_name}.png")
        img_out_path = os.path.join(images_out, f"{new_name}.png")

        PIL.Image.fromarray(lbl.astype(np.uint8), mode='L').save(mask_out_path)
        shutil.copy(image_path, img_out_path)

print("✅ Conversion complete! Check `dataset_json_to_img_ready/`.")
