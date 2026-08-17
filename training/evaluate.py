import os
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import accuracy_score, confusion_matrix
import sys
# Add current directory to path so it can find dataset and model
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'training'))
from dataset import DeepfakeDataset
from model import DeepfakeClassifier

def honest_evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_dir = "data/processed/features/test"
    
    # Load full test dataset
    full_dataset = DeepfakeDataset(root_dir=test_dir, is_train=False)
    
    # Separate Reals and Fakes based on filenames
    real_indices = [i for i, (_, lbl) in enumerate(full_dataset.images) if lbl == 0.0]
    fake_indices = [i for i, (_, lbl) in enumerate(full_dataset.images) if lbl == 1.0]
    
    real_count = len(real_indices)
    print(f"Total Reals available in test set: {real_count}")
    
    if real_count == 0:
        print("No real samples found to create balanced set.")
        return

    # Sample exactly real_count from fakes to make it 1:1
    import random
    random.seed(42)
    balanced_fake_indices = random.sample(fake_indices, real_count)
    
    balanced_indices = real_indices + balanced_fake_indices
    balanced_dataset = Subset(full_dataset, balanced_indices)
    
    test_loader = DataLoader(balanced_dataset, batch_size=8, shuffle=True)
    
    # Load model
    model = DeepfakeClassifier().to(device)
    model.load_state_dict(torch.load("models/deepfake_model.pth", map_location=device))
    model.eval()
    
    all_probs = []
    all_labels = []
    
    print(f"Running evaluation on 1:1 balanced set ({real_count} Real, {real_count} Fake)...")
    
    with torch.no_grad():
        for face_seq, lip_seq, eye_seq, mfcc, metadata, labels in test_loader:
            face_seq = face_seq.to(device)
            lip_seq = lip_seq.to(device)
            eye_seq = eye_seq.to(device)
            mfcc = mfcc.to(device)
            metadata = metadata.to(device)
            
            logits = model(face_seq, lip_seq, eye_seq, mfcc, metadata=metadata)
            probs = torch.sigmoid(logits).cpu().numpy()
            
            if probs.ndim == 0:
                probs = np.array([probs])
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
            
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Test threshold 0.1 (user default) and 0.5 (balanced)
    for t in [0.1, 0.5]:
        preds = (all_probs >= t).astype(int)
        acc = accuracy_score(all_labels, preds)
        cm = confusion_matrix(all_labels, preds)
        print(f"\n--- Honest Metrics at Threshold {t} ---")
        print(f"Balanced Accuracy: {acc*100:.2f}%")
        print(f"Confusion Matrix (Real, Fake):\n{cm}")
        print(f"Real Class Accuracy: {cm[0,0]/real_count*100:.2f}%")
        print(f"Fake Class Accuracy: {cm[1,1]/real_count*100:.2f}%")

if __name__ == "__main__":
    honest_evaluate()
