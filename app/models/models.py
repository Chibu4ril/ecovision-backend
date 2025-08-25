import torch
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor, SegformerConfig
from torchvision import models
from torchvision.models import ConvNeXt_Base_Weights
from safetensors.torch import load_file
import gdown
import os

seg_model = None
seg_processor = None
DEVICE = None
classifier = None


# Google Drive file IDs
SEGFORMER_CONFIG_ID = "1niivnfXj6z6ZiPjbE9zZXWoFs6ssTMaX"
SEGFORMER_WEIGHTS_ID = "1yIw2tIjNTYX2vgSZa_9oim0jEu4H3yPg"
CONVNEXT_MODEL_ID = "1E4QcGrJIPmcSs3ts0qhqtNKngIlVxSrP"

def download_if_missing(local_path, file_id):
    if not os.path.exists(local_path):
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        url = f"https://drive.google.com/uc?id={file_id}"
        print(f"Downloading {local_path} from Google Drive...")
        gdown.download(url, local_path, quiet=False)
        print(f"Downloaded {local_path}")


def get_models():
    global seg_model, seg_processor, DEVICE, classifier
    if seg_model is None or classifier is None:
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        seg_model, seg_processor, DEVICE = load_segformer()
        classifier = load_classifier(DEVICE)
    return seg_model, seg_processor, DEVICE, classifier

def load_segformer():
    config_path = "app/models/segformer/config.json"
    weights_path = "app/models/segformer/model.safetensors"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    download_if_missing(config_path, SEGFORMER_CONFIG_ID)
    download_if_missing(weights_path, SEGFORMER_WEIGHTS_ID)

    config = SegformerConfig.from_json_file(config_path)
    model = SegformerForSemanticSegmentation(config)
    model.load_state_dict(load_file(weights_path))
    model.to(device).eval()

    processor = SegformerImageProcessor(do_resize=True, size=640, do_normalize=True)
    return model, processor, device

def load_classifier(device):
    model_path = "app/models/convnext_classifier_base_with_coords.pth"

    download_if_missing(model_path, CONVNEXT_MODEL_ID)
    classifier = models.convnext_base(weights=ConvNeXt_Base_Weights.DEFAULT)
    classifier.classifier[2] = torch.nn.Linear(classifier.classifier[2].in_features, 2)
    classifier.load_state_dict(torch.load(model_path, map_location=device)["model_state_dict"])
    classifier.to(device).eval()
    return classifier
