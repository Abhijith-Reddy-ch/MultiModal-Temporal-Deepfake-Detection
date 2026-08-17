"""
Exp 4 (reviewer-requested): quantitative boundary-cue ablation on DFDC held-out.

The Grad-CAM spot-check in DFDC_GENERALIZATION_INVESTIGATION.md found the model's
visual attention dominated by spatial boundary/edge cues (hairline-to-background,
glasses-frame, jaw-to-background) on confidently-wrong DFDC fakes - but that was
a 6-example qualitative spot-check. This script turns it into a quantitative test:
mask/blur the face-contour boundary region (vs. interior skin, vs. an area-matched
random region) on every DFDC held-out frame and measure the resulting AUC drop.
If boundary masking hurts AUC far more than interior/random masking of the same
area, that's real evidence for boundary-cue reliance rather than a qualitative
impression from 6 images.

Never touches training data or checkpoints - eval-time-only image perturbation on
the canonical (attempt #2, see DFDC_GENERALIZATION_INVESTIGATION.md) Stage B model.
"""
import os
import sys
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'training'))
sys.path.append(os.path.join(os.getcwd(), 'pipeline'))

import argparse
import csv
import random

import cv2
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

from raw_crops_dataset import RawCropsDataset, _transform
from evaluate_stage_b import load_stage_b_models
from train_stage_b import forward_pipeline
from phase5c_sbi_augment import LandmarkDetector, derive_regions

FRAMES_PER_VIDEO = 8
BAND_PX = 12          # boundary-ring half-width (dilate/erode by this many px on a 224px crop)
BLUR_SIGMA = 12.0      # gaussian sigma for the "blurred" condition
FEATHER_SIGMA = 2.0    # feathers mask edges so interventions don't leave hard seams
BATCH_VIDEOS = 4       # videos per forward_pipeline call (each contributes 5 condition-rows)

CONDITIONS = [
    "original_reconstructed",
    "boundary_blurred",
    "boundary_masked",
    "interior_masked",
    "random_masked",
]

OUT_CSV = "outputs/predictions/dfdc_boundary_ablation.csv"


def build_hull_mask(landmarks_xy, h, w):
    mask = np.zeros((h, w), dtype=np.uint8)
    hull = cv2.convexHull(landmarks_xy.astype(np.int32))
    cv2.fillConvexPoly(mask, hull, 255)
    return mask


def band_masks(hull_mask, band_px):
    kernel = np.ones((band_px, band_px), np.uint8)
    outer = cv2.dilate(hull_mask, kernel)
    inner = cv2.erode(hull_mask, kernel)
    boundary = cv2.subtract(outer, inner)
    return boundary, inner  # boundary ring, interior


def random_region_mask(h, w, area_px, rng):
    radius = max(3, int(np.sqrt(max(area_px, 1) / np.pi)))
    radius = min(radius, min(h, w) // 2 - 1)
    cx = rng.randint(radius, w - radius)
    cy = rng.randint(radius, h - radius)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (cx, cy), radius, 255, -1)
    return mask


def feather(mask_u8, sigma=FEATHER_SIGMA):
    m = cv2.GaussianBlur(mask_u8, (0, 0), sigmaX=sigma)
    return (m.astype(np.float32) / 255.0)[..., None]


def apply_fill(face_bgr, mask_u8):
    fill = face_bgr.reshape(-1, 3).mean(axis=0)
    m = feather(mask_u8)
    filled = np.ones_like(face_bgr, dtype=np.float32) * fill
    out = m * filled + (1 - m) * face_bgr.astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_blur(face_bgr, mask_u8, sigma=BLUR_SIGMA):
    blurred = cv2.GaussianBlur(face_bgr, (0, 0), sigmaX=sigma)
    m = feather(mask_u8)
    out = m * blurred.astype(np.float32) + (1 - m) * face_bgr.astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def make_condition_face(face_bgr, condition, landmarks, rng):
    """Returns the intervened 224x224 BGR face crop for one condition, or the
    unmodified crop if landmarks is None (detection failed -> no-op fallback,
    identical across conditions for that frame, mirrors phase5c_sbi_augment's
    own fallback behavior)."""
    h, w = face_bgr.shape[:2]
    if condition == "original_reconstructed" or landmarks is None:
        return face_bgr

    hull = build_hull_mask(landmarks, h, w)
    boundary, interior = band_masks(hull, BAND_PX)

    if condition == "boundary_blurred":
        return apply_blur(face_bgr, boundary)
    if condition == "boundary_masked":
        return apply_fill(face_bgr, boundary)
    if condition == "interior_masked":
        return apply_fill(face_bgr, interior)
    if condition == "random_masked":
        area = int((boundary > 0).sum())
        rmask = random_region_mask(h, w, area, rng)
        return apply_fill(face_bgr, rmask)
    raise ValueError(condition)


def regions_to_tensor(regions_dict):
    """regions_dict: {"face":BGR224, "eyes":..., "lips":..., "jaw":...} -> stacked,
    normalized tensor in the exact frame1_face, frame1_eyes, ... order raw_crops_dataset
    expects (REGIONS = ["face","eyes","lips","jaw"])."""
    tensors = []
    for region in ["face", "eyes", "lips", "jaw"]:
        bgr = regions_dict[region]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        tensors.append(_transform(pil_img))
    return torch.stack(tensors)  # [4, 3, 224, 224]


def build_video_tensor(crop_dir, condition, detector, video_seed):
    """Builds the [32,3,224,224] image tensor for one video under one condition."""
    frame_tensors = []
    rng = random.Random(video_seed)
    for i in range(FRAMES_PER_VIDEO):
        face_path = crop_dir / f"frame{i}_face.jpg"
        face_bgr = cv2.imread(str(face_path))
        if face_bgr is None:
            face_bgr = np.zeros((224, 224, 3), dtype=np.uint8)
            landmarks = None
        else:
            landmarks = detector.landmarks_xy(face_bgr)
        cond_face = make_condition_face(face_bgr, condition, landmarks, rng)
        regions = derive_regions(cond_face)
        frame_tensors.append(regions_to_tensor(regions))  # [4,3,224,224]
    return torch.cat(frame_tensors, dim=0)  # [32,3,224,224]


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="Cap number of held_out videos (0 = all 400, for smoke-testing)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    dinov2, whisper_encoder, classifier, n_mels = load_stage_b_models(device)
    print("Canonical Stage B model loaded (attempt #2).", flush=True)

    held_out = RawCropsDataset("held_out", n_mels=n_mels, is_train=False)
    items = held_out.items
    if args.limit > 0:
        items = items[: args.limit]
    print(f"held_out size: {len(items)}", flush=True)

    detector = LandmarkDetector()

    results = {c: {"labels": [], "probs": []} for c in CONDITIONS}
    per_video_rows = []

    n = len(items)
    for start in range(0, n, BATCH_VIDEOS):
        batch_items = items[start:start + BATCH_VIDEOS]

        batch_images = []   # (num_conditions * len(batch_items), 32,3,224,224)
        batch_mel = []
        batch_meta = []
        batch_labels = []
        batch_video_ids = []

        for crop_dir, row in batch_items:
            label = 0.0 if row["label"].lower() == "real" else 1.0
            mel = held_out._load_mel(crop_dir)
            video_seed = int.from_bytes(str(crop_dir).encode("utf-8")[:8], "little", signed=False) % (2 ** 31)

            for cond_idx, condition in enumerate(CONDITIONS):
                img_tensor = build_video_tensor(crop_dir, condition, detector, video_seed + cond_idx)
                batch_images.append(img_tensor)
                batch_mel.append(mel)
                batch_meta.append(torch.zeros(34, dtype=torch.float32))
                batch_labels.append(label)
                batch_video_ids.append(str(crop_dir))

        images = torch.stack(batch_images)          # [B*5, 32,3,224,224]
        mel = torch.stack(batch_mel)                # [B*5, n_mels, 3000]
        metadata = torch.stack(batch_meta)           # [B*5, 34]

        logits, _ = forward_pipeline(dinov2, whisper_encoder, classifier, images, mel, metadata, device)
        probs = torch.sigmoid(logits).cpu().numpy().reshape(-1).tolist()

        idx = 0
        for _ in batch_items:
            for condition in CONDITIONS:
                p = probs[idx]
                lbl = batch_labels[idx]
                vid = batch_video_ids[idx]
                results[condition]["labels"].append(lbl)
                results[condition]["probs"].append(p)
                per_video_rows.append({"video": vid, "condition": condition, "label": lbl, "prob": p})
                idx += 1

        done = min(start + BATCH_VIDEOS, n)
        if done % 40 == 0 or done == n:
            print(f"Processed {done}/{n} videos...", flush=True)

    print("\n=== DFDC Boundary-Cue Ablation Results (canonical attempt #2) ===", flush=True)
    print(f"{'Condition':<24} {'AUC':>8} {'BalAcc@0.5':>12} {'n':>6}", flush=True)
    summary_rows = []
    for condition in CONDITIONS:
        labels = np.array(results[condition]["labels"])
        probs = np.array(results[condition]["probs"])
        auc = roc_auc_score(labels, probs) if len(set(labels.tolist())) > 1 else float("nan")
        bal_acc = balanced_accuracy_score(labels, (probs >= 0.5).astype(int))
        print(f"{condition:<24} {auc:>8.4f} {bal_acc:>12.4f} {len(labels):>6}", flush=True)
        summary_rows.append((condition, auc, bal_acc, len(labels)))

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["video", "condition", "label", "prob"])
        writer.writeheader()
        writer.writerows(per_video_rows)
    print(f"\nSaved per-video predictions -> {OUT_CSV}", flush=True)

    summary_csv = "outputs/predictions/dfdc_boundary_ablation_summary.csv"
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["condition", "auc", "balanced_accuracy_at_0.5", "n"])
        writer.writerows(summary_rows)
    print(f"Saved summary -> {summary_csv}", flush=True)


if __name__ == "__main__":
    main()
