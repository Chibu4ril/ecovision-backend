import os
from collections import defaultdict, Counter
import csv

# === CONFIG ===
CLASSIFIED_DIR = "outcome_folders/_07_classified_blobs_with_coords"
PATCH_SIZE_M = 4   # meters
GSD = 0.05         # meters per pixel
PATCH_PX = int(PATCH_SIZE_M / GSD)  # e.g., 40 pixels per 2m
CSV_PATH = "outcome_folders/_08_dominance_scores.csv"


# === Aggregate blobs by original image ===
blobs_by_image = defaultdict(list)

for fname in os.listdir(CLASSIFIED_DIR):
    if not fname.endswith(".txt"):
        continue

    full_path = os.path.join(CLASSIFIED_DIR, fname)

    with open(full_path, "r") as f:
        content = f.read().strip().split()
        if len(content) < 3:
            continue

        label = content[0]
        try:
            x, y, w, h = map(int, content[1:5]) if len(content) >= 5 else (0, 0, 0, 0)
        except:
            continue

        # Use the image prefix (e.g., IMG_001) as grouping key
        parts = fname.split("_")
        base_image = "_".join(parts[:2])  # e.g., IMG_001
        blobs_by_image[base_image].append((label, x, y, w, h))


# === Save results to CSV ===
with open(CSV_PATH, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Image ID", "Patch X", "Patch Y", "Dominant Species", "Percentage", "Total Blobs"])

    for image_id, blobs in blobs_by_image.items():
        patch_map = defaultdict(list)

        for label, x, y, w, h in blobs:
            center_x = x + w // 2
            center_y = y + h // 2
            patch_x = center_x // PATCH_PX
            patch_y = center_y // PATCH_PX
            patch_key = (patch_x, patch_y)
            patch_map[patch_key].append(label)

        for (px, py), labels in sorted(patch_map.items()):
            counter = Counter(labels)
            total = sum(counter.values())
            dominant = counter.most_common(1)[0]
            species, count = dominant
            percentage = (count / total) * 100
            writer.writerow([image_id, px, py, species, round(percentage, 2), total])



print("\n✅ Dominance scoring complete.")
