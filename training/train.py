"""
Stage A - Staged Training (plan.pdf Phase 5, Stage A).

Backbones (DINOv2, Whisper) are frozen and already cached to disk by
training/extract_features.py. This trains only: region projections,
cross-region attention, temporal BiLSTMs, fusion transformer, GMU, classifier,
and the auxiliary forgery-type head - all on cached features, so it's fast.

Stage B (LoRA partial unfreeze + fine-tune on raw crops/audio) is a separate
follow-up script (training/train_stage_b.py) since it needs the live
DINOv2/Whisper forward pass, not just cached tensors.
"""
import os
import re
import glob
import sys
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'training'))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from cached_dataset import CachedFeatureDataset
from model import (
    DeepfakeClassifier, MANIPULATION_TYPES, DOMAINS,
    load_classifier_state_dict, load_optimizer_state_dict,
)

CHECKPOINT_DIR = "models"
CHECKPOINT_PATTERN = os.path.join(CHECKPOINT_DIR, "stageA_checkpoint_epoch*.pth")
INPROGRESS_CKPT = os.path.join(CHECKPOINT_DIR, "stageA_checkpoint_inprogress.pth")
INPROGRESS_EVERY_N_BATCHES = 20

MAX_EPOCHS = 30
EARLY_STOP_PATIENCE = 5
AUX_LOSS_WEIGHT = 0.4  # lambda for the multi-task forgery-type loss, plan.pdf suggests 0.3-0.5

# Opt-in domain-adversarial training (DFDC investigation, attempt 7) - off by
# default so the canonical (attempt 4) recipe's behavior/reproducibility is
# unaffected unless explicitly enabled. See training/model.py's
# GradientReversalLayer for what this does.
USE_DOMAIN_ADVERSARIAL = os.environ.get("USE_DOMAIN_ADVERSARIAL", "0") == "1"
DOMAIN_ADV_LAMBDA_MAX = float(os.environ.get("DOMAIN_ADV_LAMBDA_MAX", "0.3"))


def domain_adversarial_lambda(epoch: int, max_epochs: int, lambda_max: float) -> float:
    """Sigmoid ramp-up (Ganin & Lempitsky, 2016): 0 at epoch 0, approaching
    lambda_max as training progresses, so the adversarial signal doesn't
    destabilize the trunk while its features are still uninformative early on."""
    p = epoch / max(max_epochs, 1)
    return lambda_max * (2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0)


def find_latest_checkpoint():
    ckpts = glob.glob(CHECKPOINT_PATTERN)
    if not ckpts:
        return None, 0
    def epoch_num(path):
        m = re.search(r"stageA_checkpoint_epoch(\d+)\.pth", path)
        return int(m.group(1)) if m else -1
    latest = max(ckpts, key=epoch_num)
    return latest, epoch_num(latest)


@torch.no_grad()
def evaluate_val_auc(model, val_loader, device):
    model.eval()
    all_labels, all_probs = [], []
    for visual, audio, metadata, labels, _, _ in val_loader:
        visual, audio, metadata = visual.to(device), audio.to(device), metadata.to(device)
        logits = model(visual, audio, metadata=metadata)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.extend(probs.tolist())
        all_labels.extend(labels.numpy().tolist())
    model.train()
    if len(set(all_labels)) < 2:
        return 0.5
    return roc_auc_score(all_labels, all_probs)


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dataset = CachedFeatureDataset(root_dir="data/processed/features_cached/train", is_train=True)
    val_dataset = CachedFeatureDataset(root_dir="data/processed/features_cached/val", is_train=False)

    if len(train_dataset) == 0:
        print("No cached training features found! Run training/extract_features.py first.")
        return

    num_real = sum(1 for _, label in train_dataset.items if label == 0.0)
    num_fake = sum(1 for _, label in train_dataset.items if label == 1.0)
    print(f"\n[Data Distribution] Real: {num_real}  Fake: {num_fake}  Total: {len(train_dataset)}")
    print(f"[Val set] {len(val_dataset)} videos")
    print(f"[Manipulation types] {MANIPULATION_TYPES}\n")

    pos_weight_val = num_fake / num_real if num_real > 0 else 1.0
    pos_weight_tensor = torch.tensor([pos_weight_val], dtype=torch.float32).to(device)

    sample_weights = [1.0 / num_real if label == 0.0 else 1.0 / num_fake for _, label in train_dataset.items]
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True,
    )

    batch_size = 8  # plan.pdf: small batches (4-8) shown to help Stage A, keeps memory low
    num_workers = min(4, os.cpu_count() or 1)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=sampler,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=32, shuffle=False,
        num_workers=min(2, os.cpu_count() or 1),
    )
    print(f"[DataLoader] batch_size={batch_size} num_workers={num_workers}")

    model = DeepfakeClassifier().to(device)
    criterion_bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    criterion_ce = nn.CrossEntropyLoss()
    criterion_domain = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    print(f"[Domain-Adversarial] {'ENABLED (lambda_max=' + str(DOMAIN_ADV_LAMBDA_MAX) + ')' if USE_DOMAIN_ADVERSARIAL else 'disabled'}")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    start_epoch = 1
    best_val_auc = -1.0
    epochs_without_improvement = 0

    latest_ckpt, ckpt_epoch = find_latest_checkpoint()
    if latest_ckpt is not None:
        print(f"Resuming from completed checkpoint: {latest_ckpt} (epoch {ckpt_epoch})")
        state = torch.load(latest_ckpt, map_location=device, weights_only=False)
        load_classifier_state_dict(model, state["model_state_dict"], label=latest_ckpt)
        load_optimizer_state_dict(optimizer, state["optimizer_state_dict"], label=latest_ckpt)
        best_val_auc = state.get("best_val_auc", -1.0)
        start_epoch = ckpt_epoch + 1

    if os.path.exists(INPROGRESS_CKPT):
        inprog = torch.load(INPROGRESS_CKPT, map_location=device, weights_only=False)
        if inprog["epoch"] >= start_epoch:
            print(f"Resuming from in-progress checkpoint at epoch {inprog['epoch']} "
                  f"(batch {inprog.get('batch_idx', '?')})")
            load_classifier_state_dict(model, inprog["model_state_dict"], label=INPROGRESS_CKPT)
            load_optimizer_state_dict(optimizer, inprog["optimizer_state_dict"], label=INPROGRESS_CKPT)
            start_epoch = inprog["epoch"]
            best_val_auc = inprog.get("best_val_auc", best_val_auc)

    print("Starting Stage A Training (frozen backbone, cached features)...")
    scaler = torch.amp.GradScaler(device.type)

    for epoch in range(start_epoch, MAX_EPOCHS + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        domain_correct, domain_total = 0, 0

        if USE_DOMAIN_ADVERSARIAL:
            lam = domain_adversarial_lambda(epoch, MAX_EPOCHS, DOMAIN_ADV_LAMBDA_MAX)
            model.domain_grl.set_lambda(lam)

        for batch_idx, (visual, audio, metadata, labels, manip_idx, domain_idx) in enumerate(train_loader):
            visual, audio, metadata = visual.to(device), audio.to(device), metadata.to(device)
            labels, manip_idx, domain_idx = labels.to(device), manip_idx.to(device), domain_idx.to(device)

            optimizer.zero_grad()
            with torch.amp.autocast(device.type):
                logits, feats = model(visual, audio, metadata=metadata, return_features=True)
                bce_loss = criterion_bce(logits, labels)
                ce_loss = criterion_ce(feats["aux_forgery_logits"], manip_idx)
                loss = bce_loss + AUX_LOSS_WEIGHT * ce_loss
                if USE_DOMAIN_ADVERSARIAL:
                    domain_loss = criterion_domain(feats["domain_logits"], domain_idx)
                    loss = loss + domain_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * labels.size(0)
            with torch.no_grad():
                preds = torch.sigmoid(logits) >= 0.5
                correct += (preds == labels.bool()).sum().item()
                total += labels.size(0)
                if USE_DOMAIN_ADVERSARIAL:
                    valid = domain_idx != -100
                    if valid.any():
                        domain_preds = feats["domain_logits"].argmax(dim=-1)
                        domain_correct += (domain_preds[valid] == domain_idx[valid]).sum().item()
                        domain_total += valid.sum().item()

            if (batch_idx + 1) % 10 == 0:
                extra = f" domain_ce={domain_loss.item():.4f}" if USE_DOMAIN_ADVERSARIAL else ""
                print(f"  Batch {batch_idx + 1}/{len(train_loader)} | Loss: {loss.item():.4f} "
                      f"(bce={bce_loss.item():.4f} aux_ce={ce_loss.item():.4f}{extra})")

            if (batch_idx + 1) % INPROGRESS_EVERY_N_BATCHES == 0:
                torch.save({
                    "epoch": epoch, "batch_idx": batch_idx + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_auc": best_val_auc,
                }, INPROGRESS_CKPT)

        epoch_loss = running_loss / max(total, 1)
        epoch_acc = correct / max(total, 1)
        val_auc = evaluate_val_auc(model, val_loader, device)
        msg = f"Epoch {epoch} | Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f} | Val AUC: {val_auc:.4f}"
        if USE_DOMAIN_ADVERSARIAL:
            domain_acc = domain_correct / max(domain_total, 1)
            msg += (f" | Domain Acc: {domain_acc:.4f} (chance={1/len(DOMAINS):.3f}, "
                    f"lambda={model.domain_grl.lambda_:.3f})")
        print(msg)

        ckpt_path = os.path.join(CHECKPOINT_DIR, f"stageA_checkpoint_epoch{epoch}.pth")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": epoch_loss, "accuracy": epoch_acc, "val_auc": val_auc,
            "best_val_auc": max(best_val_auc, val_auc),
        }, ckpt_path)
        print(f"Saved checkpoint: {ckpt_path}")
        if os.path.exists(INPROGRESS_CKPT):
            os.remove(INPROGRESS_CKPT)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            epochs_without_improvement = 0
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "stageA_best_model.pth"))
            print(f"  New best val AUC: {best_val_auc:.4f} - saved stageA_best_model.pth")
        else:
            epochs_without_improvement += 1
            print(f"  No improvement for {epochs_without_improvement}/{EARLY_STOP_PATIENCE} epochs")
            if epochs_without_improvement >= EARLY_STOP_PATIENCE:
                print(f"Early stopping at epoch {epoch} (best val AUC: {best_val_auc:.4f})")
                break

    print(f"Stage A Training Complete! Best model: models/stageA_best_model.pth (val AUC {best_val_auc:.4f})")


if __name__ == "__main__":
    train()
