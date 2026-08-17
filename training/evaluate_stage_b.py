"""
Evaluation for the Stage B (LoRA fine-tuned) model. Same threshold-on-val-only
discipline as training/evaluate.py, but runs the live LoRA-wrapped DINOv2 +
Whisper encoder + classifier pipeline over raw crops/audio instead of reading
pre-cached Stage A features (Stage B's backbones are no longer frozen, so
their outputs differ from the Stage A cache).
"""
import os
import sys
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'training'))

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, balanced_accuracy_score, confusion_matrix,
                             roc_auc_score, roc_curve)
from peft import PeftModel

import whisper
from raw_crops_dataset import RawCropsDataset
from model import DeepfakeClassifier, load_classifier_state_dict
from train_stage_b import forward_pipeline

CHECKPOINT_DIR = "models"


def load_stage_b_models(device):
    dinov2_base = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    dinov2 = PeftModel.from_pretrained(dinov2_base, os.path.join(CHECKPOINT_DIR, "stageB_best_dinov2_lora"))
    dinov2.eval().to(device)

    whisper_model = whisper.load_model("tiny", device=device)
    whisper_encoder = PeftModel.from_pretrained(
        whisper_model.encoder, os.path.join(CHECKPOINT_DIR, "stageB_best_whisper_lora")
    )
    whisper_encoder.eval().to(device)
    n_mels = whisper_model.dims.n_mels

    classifier = DeepfakeClassifier().to(device)
    classifier_path = os.path.join(CHECKPOINT_DIR, "stageB_best_classifier.pth")
    load_classifier_state_dict(
        classifier, torch.load(classifier_path, map_location=device, weights_only=False), label=classifier_path
    )
    classifier.eval()

    return dinov2, whisper_encoder, classifier, n_mels


def compute_metrics(all_labels, all_probs, threshold):
    preds = (np.array(all_probs) >= threshold).astype(int)
    cm = confusion_matrix(all_labels, preds, labels=[0, 1])
    acc = accuracy_score(all_labels, preds)
    precision = precision_score(all_labels, preds, zero_division=0)
    recall = recall_score(all_labels, preds, zero_division=0)
    f1 = f1_score(all_labels, preds, zero_division=0)
    return acc, precision, recall, f1, cm


@torch.no_grad()
def get_probs_labels(dinov2, whisper_encoder, classifier, loader, device):
    all_probs, all_labels = [], []
    for images, mel, metadata, labels, _, _ in loader:
        logits, _ = forward_pipeline(dinov2, whisper_encoder, classifier, images, mel, metadata, device)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.extend(np.atleast_1d(probs).tolist())
        all_labels.extend(labels.numpy().tolist())
    return np.array(all_labels), np.array(all_probs)


def pick_threshold_on_val(dinov2, whisper_encoder, classifier, val_loader, device):
    """Sweep thresholds on VALIDATION only, maximizing BALANCED accuracy (not
    F1). F1 is dominated by the majority class - on this project's ~20-85:1
    fake:real imbalance, F1-maximization pushes the threshold down toward
    predicting "fake" almost always, since that barely hurts precision on a
    fake-dominated set while maximizing fake recall. That produced a
    threshold of ~0.10 that misclassified 25-45% of real videos in practice
    despite near-perfect AUC. Balanced accuracy weights real-recall and
    fake-recall equally regardless of class counts, which is what "correctly
    classify both real and fake" actually requires."""
    val_labels, val_probs = get_probs_labels(dinov2, whisper_encoder, classifier, val_loader, device)
    if len(set(val_labels)) < 2:
        return 0.5, val_labels, val_probs
    thresholds = np.linspace(0.05, 0.99, 48)
    bal_acc_scores = [balanced_accuracy_score(val_labels, (val_probs >= t).astype(int)) for t in thresholds]
    best_idx = int(np.argmax(bal_acc_scores))
    best_threshold = thresholds[best_idx]
    print("\n--- Threshold selection on VALIDATION set (maximizing balanced accuracy) ---")
    for t, bal_acc in zip(thresholds, bal_acc_scores):
        print(f"  Threshold = {t:.2f} | Balanced Acc = {bal_acc:.4f}")
    print(f"[Chosen Threshold] {best_threshold:.3f} (Balanced Acc = {bal_acc_scores[best_idx]:.4f}), frozen for test scoring")
    return best_threshold, val_labels, val_probs


def report_table(name, labels, probs, threshold):
    if len(labels) == 0:
        print(f"\n=== {name} ===\nPENDING - no data available yet.")
        return
    acc, precision, recall, f1, cm = compute_metrics(labels, probs, threshold)
    real_total = cm[0, 0] + cm[0, 1]
    fake_total = cm[1, 0] + cm[1, 1]
    real_acc = cm[0, 0] / real_total if real_total > 0 else 0.0
    fake_acc = cm[1, 1] / fake_total if fake_total > 0 else 0.0
    print(f"\n=== {name} (threshold={threshold:.2f}, frozen from validation) ===")
    print(f"Overall Accuracy: {acc:.4f}")
    print(f"Precision:        {precision:.4f}")
    print(f"Recall:           {recall:.4f}")
    print(f"F1-score:         {f1:.4f}")
    print(f"Real Accuracy: {real_acc:.4f} ({cm[0,0]}/{real_total})   Fake Accuracy: {fake_acc:.4f} ({cm[1,1]}/{fake_total})")
    print(f"Confusion Matrix:\n{cm}")
    if len(set(labels)) > 1:
        print(f"AUC Score: {roc_auc_score(labels, probs):.4f}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dinov2, whisper_encoder, classifier, n_mels = load_stage_b_models(device)

    val_dataset = RawCropsDataset("val", n_mels=n_mels, is_train=False)
    test_dataset = RawCropsDataset("test", n_mels=n_mels, is_train=False)
    held_out_dataset = RawCropsDataset("held_out", n_mels=n_mels, is_train=False)

    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=min(4, os.cpu_count() or 1))
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=min(4, os.cpu_count() or 1))

    threshold, _, _ = pick_threshold_on_val(dinov2, whisper_encoder, classifier, val_loader, device)

    test_labels, test_probs = get_probs_labels(dinov2, whisper_encoder, classifier, test_loader, device)
    report_table("In-Distribution (FakeAVCeleb+PolyGlotFake test) - Stage B", test_labels, test_probs, threshold)

    if len(held_out_dataset) > 0:
        held_out_loader = DataLoader(held_out_dataset, batch_size=4, shuffle=False, num_workers=min(4, os.cpu_count() or 1))
        ho_labels, ho_probs = get_probs_labels(dinov2, whisper_encoder, classifier, held_out_loader, device)
        report_table("Cross-Dataset Generalization (DFDC held-out) - Stage B", ho_labels, ho_probs, threshold)
    else:
        report_table("Cross-Dataset Generalization (DFDC held-out) - Stage B", [], [], threshold)

    os.makedirs("outputs/plots", exist_ok=True)
    if len(set(test_labels)) > 1:
        fpr, tpr, _ = roc_curve(test_labels, test_probs)
        auc = roc_auc_score(test_labels, test_probs)
        plt.figure()
        plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {auc:.2f})")
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC - In-Distribution Test (Stage B)")
        plt.legend(loc="lower right")
        plt.savefig("outputs/plots/roc_curve_stageB.png")
        plt.close()
        print("\nSaved: outputs/plots/roc_curve_stageB.png")

    print("-" * 40)


if __name__ == "__main__":
    main()
