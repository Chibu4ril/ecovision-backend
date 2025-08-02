import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import convnext_base, ConvNeXt_Base_Weights
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm

# === CONFIG ===
DATA_DIR = "outcome_folders\_06_extracted_blobs_with_coords"
BATCH_SIZE = 32
NUM_CLASSES = 2
NUM_EPOCHS = 30
LEARNING_RATE = 1e-4
IMAGE_SIZE = 224
MODEL_SAVE_PATH = "trained_models/convnext_classifier_base_with_coords.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# === Dataset ===
class BlobCoordDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.class_to_idx = {}
        self.transform = transform

        for idx, class_name in enumerate(sorted(os.listdir(root_dir))):
            class_dir = os.path.join(root_dir, class_name)
            self.class_to_idx[class_name] = idx
            for fname in os.listdir(class_dir):
                if fname.endswith(".png"):
                    img_path = os.path.join(class_dir, fname)
                    txt_path = img_path.replace(".png", ".txt")
                    if os.path.exists(txt_path):
                        self.samples.append((img_path, idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

# === Transforms ===
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],  # ConvNeXt expected mean
                         [0.229, 0.224, 0.225])  # and std
])

# === Prepare Dataset & Loader ===
dataset = BlobCoordDataset(DATA_DIR, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
class_names = list(dataset.class_to_idx.keys())
print("Classes:", class_names)

# === Load Pretrained ConvNeXt-Base ===
weights = ConvNeXt_Base_Weights.DEFAULT  
model = convnext_base(weights=weights)
model.classifier[2] = nn.Linear(model.classifier[2].in_features, NUM_CLASSES)
model = model.to(device)

# === Loss & Optimizer ===
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# === Training Loop ===
for epoch in range(NUM_EPOCHS):
    model.train()
    train_loss, correct = 0, 0
    for images, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()

    acc = correct / len(dataset)
    print(f"[Epoch {epoch+1}] Loss: {train_loss:.4f} | Accuracy: {acc:.4f}")

# === Save Model ===
torch.save({
    'model_state_dict': model.state_dict(),
    'class_names': class_names,
    'image_size': IMAGE_SIZE
}, MODEL_SAVE_PATH)

print(f"✅ ConvNeXt-Base classifier trained and saved to {MODEL_SAVE_PATH}")
