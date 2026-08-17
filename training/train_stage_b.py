"""
Stage B - Staged Training (plan.pdf Phase 5, Stage B).

Adds LoRA adapters to the last 4 DINOv2 transformer blocks and the last
Whisper-tiny encoder block, then fine-tunes them end-to-end alongside the
Stage A head (cross-region attention, BiLSTMs, fusion transformer, GMU,
classifier, aux head) at a much lower LR. Starts from stageA_best_model.pth.

Needs raw crops/audio (not cached features), since the backbones are no
longer frozen - runs slower than Stage A, uses gradient accumulation to
reach an effective batch of 16-32 while keeping the physical batch small
enough for 8GB VRAM.
"""
import os
import re
import glob
import random
import sys
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'training'))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from peft import LoraConfig, get_peft_model

import math

import whisper
from raw_crops_dataset import RawCropsDataset
from model import (
    DeepfakeClassifier, MANIPULATION_TYPES, DOMAINS,
    load_classifier_state_dict, load_optimizer_state_dict,
)

CHECKPOINT_DIR = "models"
CHECKPOINT_PATTERN = os.path.join(CHECKPOINT_DIR, "stageB_checkpoint_epoch*.pth")
INPROGRESS_CKPT = os.path.join(CHECKPOINT_DIR, "stageB_checkpoint_inprogress.pth")
INPROGRESS_EVERY_N_BATCHES = 20

MAX_EPOCHS = 15
EARLY_STOP_PATIENCE = 4
AUX_LOSS_WEIGHT = 0.4

# Opt-in domain-adversarial training (DFDC investigation, attempt 7) - off by
# default so the canonical (attempt 4) recipe's behavior/reproducibility is
# unaffected unless explicitly enabled. See training/model.py's
# GradientReversalLayer for what this does.
USE_DOMAIN_ADVERSARIAL = os.environ.get("USE_DOMAIN_ADVERSARIAL", "0") == "1"
DOMAIN_ADV_LAMBDA_MAX = float(os.environ.get("DOMAIN_ADV_LAMBDA_MAX", "0.3"))


def domain_adversarial_lambda(epoch: int, max_epochs: int, lambda_max: float) -> float:
    """Sigmoid ramp-up (Ganin & Lempitsky, 2016) - see training/train.py's
    identical helper for the rationale."""
    p = epoch / max(max_epochs, 1)
    return lambda_max * (2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0)

PHYSICAL_BATCH = 2
GRAD_ACCUM_STEPS = 8  # effective batch = 16 - reverted after measuring: batch=4/workers=10
                      # was actually SLOWER in practice (538 samples/min vs 742), likely
                      # DataLoader worker contention outweighing the larger-batch benefit
FRAMES_PER_VIDEO = 8
NUM_REGIONS = 4
DINO_DIM = 384

DINO_LORA_BLOCKS = [8, 9, 10, 11]   # last 4 of DINOv2 ViT-S/14's 12 blocks
WHISPER_LORA_BLOCKS = [3]           # last 1 of Whisper-tiny's 4 encoder blocks

# Opt-in Self-Blended Images augmentation (pipeline/phase5c_sbi_augment.py) -
# off by default so existing training behavior/reproducibility is unaffected
# until explicitly enabled.
USE_SBI_AUGMENTATION = os.environ.get("USE_SBI_AUGMENTATION", "0") == "1"

# Both default to the attempt-4 recipe (augmentation on, weight_decay=1e-4)
# that was hardcoded/unconditional here historically - added 2026-08-16 so a
# second retrain can reproduce attempt #2's exact recipe (no augmentation, no
# weight decay) on the corrected identity-safe splits, per
# DFDC_GENERALIZATION_INVESTIGATION.md's post-identity-fix retrain section.
USE_AUGMENTATION = os.environ.get("USE_AUGMENTATION", "1") == "1"
WEIGHT_DECAY = float(os.environ.get("WEIGHT_DECAY", "1e-4"))


def build_dinov2_with_lora(device):
    dinov2 = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    target_modules = [
        f"blocks.{b}.{layer}"
        for b in DINO_LORA_BLOCKS
        for layer in ["attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"]
    ]
    config = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.1, target_modules=target_modules, bias="none")
    dinov2 = get_peft_model(dinov2, config)
    dinov2.to(device)
    return dinov2


def build_whisper_encoder_with_lora(device):
    whisper_model = whisper.load_model("tiny", device=device)
    encoder = whisper_model.encoder
    target_modules = [
        f"blocks.{b}.{layer}"
        for b in WHISPER_LORA_BLOCKS
        for layer in ["attn.query", "attn.key", "attn.value", "attn.out", "mlp.0", "mlp.2"]
    ]
    config = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.1, target_modules=target_modules, bias="none")
    encoder = get_peft_model(encoder, config)
    encoder.to(device)
    return encoder, whisper_model.dims.n_mels


def find_latest_checkpoint():
    ckpts = glob.glob(CHECKPOINT_PATTERN)
    if not ckpts:
        return None, 0
    def epoch_num(path):
        m = re.search(r"stageB_checkpoint_epoch(\d+)\.pth", path)
        return int(m.group(1)) if m else -1
    latest = max(ckpts, key=epoch_num)
    return latest, epoch_num(latest)


def forward_pipeline(dinov2, whisper_encoder, classifier, images, mel, metadata, device):
    """images: (B,32,3,224,224), mel: (B,n_mels,3000) -> logits, aux_logits"""
    B = images.shape[0]
    images_flat = images.view(B * FRAMES_PER_VIDEO * NUM_REGIONS, 3, 224, 224).to(device)
    mel = mel.to(device)

    cls_tokens = dinov2(images_flat)
    visual = cls_tokens.view(B, FRAMES_PER_VIDEO, NUM_REGIONS, DINO_DIM)

    hidden = whisper_encoder(mel)  # (B, 1500, 384)
    hidden = hidden.transpose(1, 2)
    audio = F.adaptive_avg_pool1d(hidden, FRAMES_PER_VIDEO).transpose(1, 2)  # (B, 8, 384)

    metadata = metadata.to(device)
    logits, feats = classifier(visual, audio, metadata=metadata, return_features=True)
    return logits, feats


@torch.no_grad()
def evaluate_val_auc(dinov2, whisper_encoder, classifier, val_loader, device):
    classifier.eval()
    all_labels, all_probs = [], []
    for images, mel, metadata, labels, _, _ in val_loader:
        logits, _ = forward_pipeline(dinov2, whisper_encoder, classifier, images, mel, metadata, device)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.extend(probs.tolist())
        all_labels.extend(labels.numpy().tolist())
    classifier.train()
    if len(set(all_labels)) < 2:
        return 0.5
    return roc_auc_score(all_labels, all_probs)


def train():
    # Opt-in global seed (2026-08-17, multiple-seed variance study per reviewer
    # recommendation) - unset by default so pre-existing runs' behavior/
    # reproducibility notes ("no global torch seed fixed for weight init or
    # training-loop stochasticity") are unaffected unless explicitly requested.
    seed_env = os.environ.get("SEED")
    if seed_env is not None:
        seed = int(seed_env)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        print(f"[Seed] {seed} (torch/numpy/random all seeded)")
    else:
        print("[Seed] none set (default nondeterministic weight init/training-loop stochasticity)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading DINOv2 ViT-S/14 with LoRA on last 4 blocks...")
    dinov2 = build_dinov2_with_lora(device)
    dinov2.print_trainable_parameters()

    print("Loading Whisper-tiny encoder with LoRA on last block...")
    whisper_encoder, n_mels = build_whisper_encoder_with_lora(device)
    whisper_encoder.print_trainable_parameters()

    classifier = DeepfakeClassifier().to(device)
    stage_a_path = os.path.join(CHECKPOINT_DIR, "stageA_best_model.pth")
    if os.path.exists(stage_a_path):
        print(f"Loading Stage A weights from {stage_a_path}")
        load_classifier_state_dict(
            classifier, torch.load(stage_a_path, map_location=device, weights_only=False), label=str(stage_a_path)
        )
    else:
        print("WARNING: no Stage A checkpoint found - starting classifier head from scratch")

    train_dataset = RawCropsDataset("train", n_mels=n_mels, is_train=True, use_sbi=USE_SBI_AUGMENTATION,
                                     use_augmentation=USE_AUGMENTATION)
    val_dataset = RawCropsDataset("val", n_mels=n_mels, is_train=False)
    if len(train_dataset) == 0:
        print("No training crops found! Run pipeline/phase5_extract.py first.")
        return

    num_real = sum(1 for _, row in train_dataset.items if row["label"].lower() == "real")
    num_sbi = sum(1 for _, row in train_dataset.items if row.get("source") == "SBI")
    num_fake = len(train_dataset) - num_real
    print(f"\n[Data Distribution] Real: {num_real}  Fake: {num_fake} (of which SBI: {num_sbi})  Total: {len(train_dataset)}")
    print(f"[SBI Augmentation] {'ENABLED' if USE_SBI_AUGMENTATION else 'disabled'}")
    print(f"[Video Augmentation] {'ENABLED' if USE_AUGMENTATION else 'disabled'}")
    print(f"[Weight Decay] {WEIGHT_DECAY}")
    print(f"[Val set] {len(val_dataset)} videos\n")

    pos_weight_val = num_fake / num_real if num_real > 0 else 1.0
    pos_weight_tensor = torch.tensor([pos_weight_val], dtype=torch.float32).to(device)

    sample_weights = [
        1.0 / num_real if row["label"].lower() == "real" else 1.0 / num_fake
        for _, row in train_dataset.items
    ]
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True,
    )

    num_workers = min(4, os.cpu_count() or 1)  # settled here for good - workers=6 crashed with
                                                 # WinError 1455 (pagefile exhausted), workers=10
                                                 # was just slower. This machine's real ceiling is 4.
    train_loader = DataLoader(
        train_dataset, batch_size=PHYSICAL_BATCH, sampler=sampler,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=min(6, os.cpu_count() or 1))
    print(f"[DataLoader] physical_batch={PHYSICAL_BATCH} grad_accum={GRAD_ACCUM_STEPS} "
          f"(effective batch={PHYSICAL_BATCH * GRAD_ACCUM_STEPS}) num_workers={num_workers}")

    lora_params = [p for p in dinov2.parameters() if p.requires_grad] + \
                  [p for p in whisper_encoder.parameters() if p.requires_grad]
    head_params = list(classifier.parameters())

    # weight_decay added as a defense against overfitting to the training
    # datasets' specific artifacts (augmentation alone didn't close the
    # cross-dataset generalization gap - see project memory for the full
    # comparison across attempts).
    optimizer = optim.Adam([
        {"params": lora_params, "lr": 1e-5, "weight_decay": WEIGHT_DECAY},
        {"params": head_params, "lr": 1e-4, "weight_decay": WEIGHT_DECAY},
    ])
    criterion_bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    criterion_ce = nn.CrossEntropyLoss()
    criterion_domain = nn.CrossEntropyLoss(ignore_index=-100)

    print(f"[Domain-Adversarial] {'ENABLED (lambda_max=' + str(DOMAIN_ADV_LAMBDA_MAX) + ')' if USE_DOMAIN_ADVERSARIAL else 'disabled'}")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    start_epoch = 1
    best_val_auc = -1.0
    epochs_without_improvement = 0

    latest_ckpt, ckpt_epoch = find_latest_checkpoint()
    if latest_ckpt is not None:
        print(f"Resuming from completed checkpoint: {latest_ckpt} (epoch {ckpt_epoch})")
        state = torch.load(latest_ckpt, map_location=device, weights_only=False)
        dinov2.load_state_dict(state["dinov2_state_dict"])
        whisper_encoder.load_state_dict(state["whisper_encoder_state_dict"])
        load_classifier_state_dict(classifier, state["classifier_state_dict"], label=latest_ckpt)
        load_optimizer_state_dict(optimizer, state["optimizer_state_dict"], label=latest_ckpt)
        best_val_auc = state.get("best_val_auc", -1.0)
        start_epoch = ckpt_epoch + 1

    if os.path.exists(INPROGRESS_CKPT):
        inprog = torch.load(INPROGRESS_CKPT, map_location=device, weights_only=False)
        if inprog["epoch"] >= start_epoch:
            print(f"Resuming from in-progress checkpoint at epoch {inprog['epoch']} "
                  f"(batch {inprog.get('batch_idx', '?')})")
            dinov2.load_state_dict(inprog["dinov2_state_dict"])
            whisper_encoder.load_state_dict(inprog["whisper_encoder_state_dict"])
            load_classifier_state_dict(classifier, inprog["classifier_state_dict"], label=INPROGRESS_CKPT)
            load_optimizer_state_dict(optimizer, inprog["optimizer_state_dict"], label=INPROGRESS_CKPT)
            start_epoch = inprog["epoch"]
            best_val_auc = inprog.get("best_val_auc", best_val_auc)

    if start_epoch > MAX_EPOCHS:
        print("Stage B training already completed per checkpoints. Exiting.")
        return

    print("Starting Stage B Training (LoRA fine-tune, raw crops/audio)...")
    scaler = torch.amp.GradScaler(device.type)

    for epoch in range(start_epoch, MAX_EPOCHS + 1):
        classifier.train()
        dinov2.train()
        whisper_encoder.train()

        running_loss, correct, total = 0.0, 0, 0
        domain_correct, domain_total = 0, 0
        optimizer.zero_grad()

        if USE_DOMAIN_ADVERSARIAL:
            lam = domain_adversarial_lambda(epoch, MAX_EPOCHS, DOMAIN_ADV_LAMBDA_MAX)
            classifier.domain_grl.set_lambda(lam)

        for batch_idx, (images, mel, metadata, labels, manip_idx, domain_idx) in enumerate(train_loader):
            labels, manip_idx, domain_idx = labels.to(device), manip_idx.to(device), domain_idx.to(device)

            with torch.amp.autocast(device.type):
                logits, feats = forward_pipeline(dinov2, whisper_encoder, classifier, images, mel, metadata, device)
                bce_loss = criterion_bce(logits, labels)
                ce_loss = criterion_ce(feats["aux_forgery_logits"], manip_idx)
                total_loss = bce_loss + AUX_LOSS_WEIGHT * ce_loss
                if USE_DOMAIN_ADVERSARIAL:
                    domain_loss = criterion_domain(feats["domain_logits"], domain_idx)
                    total_loss = total_loss + domain_loss
                loss = total_loss / GRAD_ACCUM_STEPS

            scaler.scale(loss).backward()

            if (batch_idx + 1) % GRAD_ACCUM_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            running_loss += loss.item() * GRAD_ACCUM_STEPS * labels.size(0)
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

            if (batch_idx + 1) % 40 == 0:
                extra = f" domain_ce={domain_loss.item():.4f}" if USE_DOMAIN_ADVERSARIAL else ""
                print(f"  Batch {batch_idx + 1}/{len(train_loader)} | Loss: {loss.item() * GRAD_ACCUM_STEPS:.4f}{extra}")

            if (batch_idx + 1) % INPROGRESS_EVERY_N_BATCHES == 0:
                torch.save({
                    "epoch": epoch, "batch_idx": batch_idx + 1,
                    "dinov2_state_dict": dinov2.state_dict(),
                    "whisper_encoder_state_dict": whisper_encoder.state_dict(),
                    "classifier_state_dict": classifier.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_auc": best_val_auc,
                }, INPROGRESS_CKPT)

        epoch_loss = running_loss / max(total, 1)
        epoch_acc = correct / max(total, 1)
        val_auc = evaluate_val_auc(dinov2, whisper_encoder, classifier, val_loader, device)
        msg = f"Epoch {epoch} | Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f} | Val AUC: {val_auc:.4f}"
        if USE_DOMAIN_ADVERSARIAL:
            domain_acc = domain_correct / max(domain_total, 1)
            msg += (f" | Domain Acc: {domain_acc:.4f} (chance={1/len(DOMAINS):.3f}, "
                    f"lambda={classifier.domain_grl.lambda_:.3f})")
        print(msg)

        ckpt_path = os.path.join(CHECKPOINT_DIR, f"stageB_checkpoint_epoch{epoch}.pth")
        torch.save({
            "epoch": epoch,
            "dinov2_state_dict": dinov2.state_dict(),
            "whisper_encoder_state_dict": whisper_encoder.state_dict(),
            "classifier_state_dict": classifier.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_auc": val_auc, "best_val_auc": max(best_val_auc, val_auc),
        }, ckpt_path)
        print(f"Saved checkpoint: {ckpt_path}")
        if os.path.exists(INPROGRESS_CKPT):
            os.remove(INPROGRESS_CKPT)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            epochs_without_improvement = 0
            torch.save(classifier.state_dict(), os.path.join(CHECKPOINT_DIR, "stageB_best_classifier.pth"))
            dinov2.save_pretrained(os.path.join(CHECKPOINT_DIR, "stageB_best_dinov2_lora"))
            whisper_encoder.save_pretrained(os.path.join(CHECKPOINT_DIR, "stageB_best_whisper_lora"))
            print(f"  New best val AUC: {best_val_auc:.4f} - saved stageB_best_* artifacts")
        else:
            epochs_without_improvement += 1
            print(f"  No improvement for {epochs_without_improvement}/{EARLY_STOP_PATIENCE} epochs")
            if epochs_without_improvement >= EARLY_STOP_PATIENCE:
                print(f"Early stopping at epoch {epoch} (best val AUC: {best_val_auc:.4f})")
                break

    print(f"Stage B Training Complete! Best val AUC: {best_val_auc:.4f}")


if __name__ == "__main__":
    train()
