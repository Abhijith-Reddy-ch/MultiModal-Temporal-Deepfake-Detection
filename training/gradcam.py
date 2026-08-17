import argparse
from pathlib import Path
from PIL import Image
import torch
import cv2
import numpy as float32
import numpy as np

# Require library: pip install grad-cam
try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
except ImportError:
    print("Please install grad-cam: pip install grad-cam")
    exit(1)

from training.model import DeepfakeClassifier
from training.config import MODELS_DIR, PLOTS_DIR
from training.infer import load_model, get_infer_transforms
import torch.nn as nn

class GradCamWrapper(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        
    def forward(self, face_tensor):
        # face_tensor: [B, C, H, W]
        device = face_tensor.device
        f_s = face_tensor.unsqueeze(1) # [B, 1, C, H, W]
        l_s = torch.zeros_like(f_s)
        e_s = torch.zeros_like(f_s)
        m_s = torch.zeros(f_s.size(0), 1, 40, 128).to(device)
        logits = self.base_model(f_s, l_s, e_s, m_s)
        # GradCAM expects [Batch, Classes], so we unsqueeze the 1D output
        return logits.unsqueeze(-1)

def run_gradcam(image_path: str):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image_path = Path(image_path)
    img_pil = Image.open(image_path).convert('RGB')
    
    # Keep original for visualization
    img_vis = cv2.imread(str(image_path))
    img_vis = cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB)
    img_vis = cv2.resize(img_vis, (224, 224))
    img_vis_norm = np.float32(img_vis) / 255
    
    transform = get_infer_transforms()
    input_tensor = transform(img_pil).unsqueeze(0).to(device)
    
    base_model = load_model()
    model = GradCamWrapper(base_model).to(device)
    model.eval()
    
    # The last conv layer in ResNet (layer4) is at index 7 of features
    target_layers = [base_model.face_branch.encoder.features[7][-1]]
    
    cam = GradCAM(model=model, target_layers=target_layers)
    
    # We use ClassifierOutputTarget(0) since we only have 1 active output neuron (binary classification)
    targets = [ClassifierOutputTarget(0)]
    
    # Generate heatmap - disable cuDNN to allow RNN backward in eval mode
    with torch.backends.cudnn.flags(enabled=False):
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        
    grayscale_cam = grayscale_cam[0, :]
    
    visualization = show_cam_on_image(img_vis_norm, grayscale_cam, use_rgb=True)
    
    out_path = PLOTS_DIR / f"gradcam_{image_path.name}"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert rgb to bgr for cv2 save
    cv2.imwrite(str(out_path), cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
    print(f"Grad-CAM visualization saved to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grad-CAM DeepFake Visualizer")
    parser.add_argument("image", type=str, help="Path to the image to analyze")
    args = parser.parse_args()
    run_gradcam(args.image)
