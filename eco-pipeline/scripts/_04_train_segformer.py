import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor, TrainingArguments, Trainer
import evaluate
from natsort import natsorted

# === CONFIG ===
NUM_CLASSES = 3  # background, spartina, puccinellia
DATA_DIR = "outcome_folders\_04_final_split_dataset"
MODEL_NAME = "nvidia/segformer-b5-finetuned-ade-640-640"
OUTPUT_DIR = "trained_models\segformer"
BATCH_SIZE = 2
IMAGE_SIZE = 512

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")

# === Preprocessor ===
image_processor = SegformerImageProcessor(do_resize=True, size=IMAGE_SIZE, do_normalize=True)

# === Custom Dataset ===
class SegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir):
        self.images = natsorted([os.path.join(images_dir, f) for f in os.listdir(images_dir) if f.endswith(".png")])
        self.masks = natsorted([os.path.join(masks_dir, f) for f in os.listdir(masks_dir) if f.endswith(".png")])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert("RGB")
        mask = Image.open(self.masks[idx])
        image = np.array(image)
        mask = np.array(mask.resize((IMAGE_SIZE, IMAGE_SIZE), resample=Image.NEAREST)).astype(np.uint8)

        inputs = image_processor(image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].squeeze(0)
        return {
            "pixel_values": pixel_values,
            "labels": torch.tensor(mask, dtype=torch.long)
        }

# === Load Dataset Splits ===
train_dataset = SegmentationDataset(f"{DATA_DIR}/train/images", f"{DATA_DIR}/train/masks")
val_dataset = SegmentationDataset(f"{DATA_DIR}/val/images", f"{DATA_DIR}/val/masks")

# === Model ===
model = SegformerForSemanticSegmentation.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_CLASSES,
    ignore_mismatched_sizes=True,
)

# === Metrics ===
metric = evaluate.load("mean_iou")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    resized_preds = []
    resized_labels = []

    for p, l in zip(preds, labels):
        if p.shape != l.shape:
            p = Image.fromarray(p.astype(np.uint8))
            p = p.resize(l.shape[::-1], resample=Image.NEAREST)
            p = np.array(p)
        resized_preds.append(p)
        resized_labels.append(l)

    resized_preds = np.array(resized_preds)
    resized_labels = np.array(resized_labels)

    raw_metrics = metric.compute(
        predictions=resized_preds,
        references=resized_labels,
        num_labels=NUM_CLASSES,
        ignore_index=0
    )

    # Convert all NumPy types to Python scalars or lists
    clean_metrics = {}
    for k, v in raw_metrics.items():
        if isinstance(v, np.ndarray):
            if v.size == 1:
                clean_metrics[k] = float(v)
            else:
                clean_metrics[k] = v.tolist()  # convert arrays to lists
        else:
            clean_metrics[k] = float(v) if isinstance(v, (np.float32, np.float64)) else v

    print("Eval Metrics:", clean_metrics)
    return clean_metrics



# === Training Arguments ===
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    eval_strategy="steps",  # NOTE: renamed from evaluation_strategy in new versions
    save_steps=1000,
    eval_steps=500,
    num_train_epochs=10,
    logging_dir=f"{OUTPUT_DIR}/logs",
    logging_steps=50,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="mean_iou",
    remove_unused_columns=False,
    report_to="none",
    fp16=True,
)

# === Trainer ===
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

# === Start Training ===
# trainer.train(resume_from_checkpoint='trained_models\segformer')
trainer.train()
trainer.save_model(OUTPUT_DIR)
print("✅ SegFormer training complete. Model saved to", OUTPUT_DIR)
