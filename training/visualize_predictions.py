import os
import random
import torch
import matplotlib.pyplot as plt
from PIL import Image
from training.dataset import DeepfakeDataset
from training.model import DeepfakeClassifier

def visualize_samples(num_samples=16):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    test_dir = "data/processed/features/test"
    test_dataset = DeepfakeDataset(root_dir=test_dir, is_train=False)
    
    if len(test_dataset) == 0:
        print("No images found in test dataset.")
        return
        
    model_path = "models/deepfake_model.pth"
    if not os.path.exists(model_path):
        print("Model not found. Please train first.")
        return
        
    model = DeepfakeClassifier()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    os.makedirs("outputs/predictions", exist_ok=True)
    
    indices = random.sample(range(len(test_dataset)), min(num_samples, len(test_dataset)))
    
    cols = 4
    rows = (len(indices) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows))
    axes = axes.flatten()
    
    for i, idx in enumerate(indices):
        img_path, true_label = test_dataset.images[idx]
        
        # Original Image for display
        img_display = Image.open(img_path).convert("RGB")
        
        # Tensors for model
        face_seq, lip_seq, eye_seq, mfcc, metadata_tensor, _ = test_dataset[idx]
        face_seq = face_seq.unsqueeze(0).to(device)
        lip_seq  = lip_seq.unsqueeze(0).to(device)
        eye_seq  = eye_seq.unsqueeze(0).to(device)
        mfcc     = mfcc.unsqueeze(0).to(device)
        metadata_tensor = metadata_tensor.unsqueeze(0).to(device)
        
        with torch.no_grad():
            logit = model(face_seq, lip_seq, eye_seq, mfcc, metadata=metadata_tensor)
            prob = torch.sigmoid(logit.squeeze(-1)).item()
            
        pred = 1 if prob >= 0.5 else 0
        conf = prob if pred == 1 else (1 - prob)
        
        true_str = "Fake" if true_label == 1.0 else "Real"
        pred_str = "Fake" if pred == 1 else "Real"
        
        color = "green" if pred == int(true_label) else "red"
        
        axes[i].imshow(img_display)
        axes[i].axis("off")
        axes[i].set_title(f"True: {true_str}\nPred: {pred_str} ({conf:.2f})", color=color)
        
    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
        
    plt.tight_layout()
    out_path = "outputs/predictions/sample_predictions.png"
    plt.savefig(out_path)
    print(f"Visualization saved to {out_path}")

if __name__ == "__main__":
    visualize_samples()
