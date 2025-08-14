import torch
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor, SegformerConfig
from torchvision import models
from torchvision.models import ConvNeXt_Base_Weights
from safetensors.torch import load_file

def load_segformer():
    config_path = "app/models/segformer/config.json"
    weights_path = "app/models/segformer/model.safetensors"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = SegformerConfig.from_json_file(config_path)
    model = SegformerForSemanticSegmentation(config)
    model.load_state_dict(load_file(weights_path))
    model.to(device).eval()

    processor = SegformerImageProcessor(do_resize=True, size=640, do_normalize=True)
    return model, processor, device

def load_classifier(device):
    model_path = "app/models/convnext_classifier_base_with_coords.pth"
    classifier = models.convnext_base(weights=ConvNeXt_Base_Weights.DEFAULT)
    classifier.classifier[2] = torch.nn.Linear(classifier.classifier[2].in_features, 2)
    classifier.load_state_dict(torch.load(model_path, map_location=device)["model_state_dict"])
    classifier.to(device).eval()
    return classifier
