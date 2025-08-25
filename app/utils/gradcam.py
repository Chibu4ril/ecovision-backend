import torch
import numpy as np
import cv2
from PIL import Image

def gradcam_full_image(model, input_tensor, target_class: int, last_conv_layer, orig_pil_img, bbox):
    gradients = []
    activations = []

    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    handle_forward = last_conv_layer.register_forward_hook(forward_hook)
    handle_backward = last_conv_layer.register_full_backward_hook(backward_hook)

    model.zero_grad()
    output = model(input_tensor)
    loss = output[0, target_class]
    loss.backward()

    handle_forward.remove()
    handle_backward.remove()

    acts = activations[0][0].detach().cpu()   # [C, H, W]
    grads = gradients[0][0].detach().cpu()    # [C, H, W]

    weights = grads.mean(dim=(1, 2))          # [C]
    cam = (weights[:, None, None] * acts).sum(dim=0)  # [H, W]
    cam = torch.relu(cam)
    cam = (cam - cam.min()) / (cam.max() + 1e-8)

    x, y, w_box, h_box = bbox
    cam_resized = torch.nn.functional.interpolate(
        cam.unsqueeze(0).unsqueeze(0),
        size=(int(h_box), int(w_box)),
        mode='bilinear',
        align_corners=False
    )[0, 0]

    heatmap = np.uint8(255 * cam_resized.numpy())
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    orig_img = np.array(orig_pil_img)
    blended = orig_img.copy()
    blended[y:y+h_box, x:x+w_box] = cv2.addWeighted(orig_img[y:y+h_box, x:x+w_box], 0.5, heatmap, 0.5, 0)

    return Image.fromarray(blended)
