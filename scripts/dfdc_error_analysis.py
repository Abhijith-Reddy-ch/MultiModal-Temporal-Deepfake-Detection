"""One-off diagnostic (no retraining): break down the current canonical Stage
B model's DFDC held-out predictions by technical video properties (
resolution/fps/bitrate/codec/duration) to see whether misclassifications
cluster around specific technical characteristics rather than being uniform -
this determines whether the generalization gap is about content/generator
diversity (need new data) or a covariate-shift artifact (need targeted
augmentation) before committing to either fix.
"""
import os
import sys
import json
import csv
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'training'))

import numpy as np
import torch
from torch.utils.data import DataLoader

from raw_crops_dataset import RawCropsDataset
from evaluate_stage_b import load_stage_b_models, forward_pipeline

sys.path.append(os.path.join(os.getcwd(), 'scripts'))
from extract_media_metadata import process_single_video

# From project memory: this exact weight-decay-winner checkpoint's val-derived
# balanced-accuracy threshold (VAL AUC 0.9973, best threshold 0.940). Reused
# here rather than re-swept, since this is a fast diagnostic over the SAME
# restored canonical checkpoint, not a new training run.
THRESHOLD = 0.940


@torch.no_grad()
def get_probs_ordered(dinov2, whisper_encoder, classifier, dataset, device, batch_size=4):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=min(4, os.cpu_count() or 1))
    all_probs = []
    for images, mel, metadata, labels, _, _ in loader:
        logits, _ = forward_pipeline(dinov2, whisper_encoder, classifier, images, mel, metadata, device)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.extend(np.atleast_1d(probs).tolist())
    return all_probs


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dinov2, whisper_encoder, classifier, n_mels = load_stage_b_models(device)

    held_out_dataset = RawCropsDataset("held_out", n_mels=n_mels, is_train=False)
    print(f"held_out dataset size: {len(held_out_dataset)}")

    probs = get_probs_ordered(dinov2, whisper_encoder, classifier, held_out_dataset, device)
    assert len(probs) == len(held_out_dataset.items)

    dfdc_meta = {}
    meta_path = "data/raw/DFDC/metadata.json"
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            dfdc_meta = json.load(f)

    rows = []
    for (crop_dir, row), prob in zip(held_out_dataset.items, probs):
        video_path = row["video_path"]
        label = row["label"].lower()
        fname = os.path.basename(video_path)
        dm = dfdc_meta.get(fname, {})
        rows.append({
            "video_path": video_path,
            "fname": fname,
            "label": label,
            "prob": prob,
            "pred": "fake" if prob >= THRESHOLD else "real",
            "correct": (prob >= THRESHOLD) == (label == "fake"),
            "original": dm.get("original"),
        })

    print(f"\nCollected {len(rows)} predictions. Extracting ffprobe technical metadata (this takes a bit)...")
    for r in rows:
        _, tm = process_single_video(r["video_path"])
        if tm:
            r.update({
                "width": tm["width"], "height": tm["height"], "fps": round(tm["fps"], 2),
                "duration": round(tm["duration"], 2), "video_bitrate": tm["video_bitrate"],
                "video_codec": tm["video_codec"], "nb_frames": tm["nb_frames"],
            })

    out_csv = "outputs/predictions/dfdc_error_analysis.csv"
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved per-video breakdown: {out_csv}")

    # ---- Summary ----
    import statistics as stats

    def summarize(subset, name):
        if not subset:
            print(f"{name}: (empty)")
            return
        widths = [r["width"] for r in subset if r.get("width")]
        heights = [r["height"] for r in subset if r.get("height")]
        fps_l = [r["fps"] for r in subset if r.get("fps")]
        durs = [r["duration"] for r in subset if r.get("duration")]
        brs = [r["video_bitrate"] for r in subset if r.get("video_bitrate")]
        codecs = {}
        for r in subset:
            c = r.get("video_codec", "?")
            codecs[c] = codecs.get(c, 0) + 1
        print(f"\n--- {name} (n={len(subset)}) ---")
        if widths:
            print(f"  resolution: {stats.median(widths)}x{stats.median(heights)} (median)  "
                  f"range w=[{min(widths)},{max(widths)}] h=[{min(heights)},{max(heights)}]")
        if fps_l:
            print(f"  fps: median={stats.median(fps_l):.2f} range=[{min(fps_l):.2f},{max(fps_l):.2f}]")
        if durs:
            print(f"  duration: median={stats.median(durs):.2f}s")
        if brs:
            print(f"  video_bitrate: median={stats.median(brs):.0f}")
        print(f"  codecs: {codecs}")
        probs_sub = [r["prob"] for r in subset]
        print(f"  prob: median={stats.median(probs_sub):.4f} mean={stats.mean(probs_sub):.4f}")

    reals = [r for r in rows if r["label"] == "real"]
    fakes = [r for r in rows if r["label"] == "fake"]
    real_correct = [r for r in reals if r["correct"]]
    real_wrong = [r for r in reals if not r["correct"]]
    fake_correct = [r for r in fakes if r["correct"]]
    fake_wrong = [r for r in fakes if not r["correct"]]

    print("\n" + "=" * 60)
    print(f"THRESHOLD USED: {THRESHOLD}")
    print(f"Overall: {len(rows)} videos | Real: {len(reals)} | Fake: {len(fakes)}")
    print(f"Real: {len(real_correct)}/{len(reals)} correct | Fake: {len(fake_correct)}/{len(fakes)} correct")

    summarize(real_correct, "REAL - correctly classified")
    summarize(real_wrong, "REAL - misclassified as fake")
    summarize(fake_correct, "FAKE - correctly classified")
    summarize(fake_wrong, "FAKE - misclassified as real (the big failure mode)")

    # Check whether missed fakes share "original" source videos disproportionately
    from collections import Counter
    wrong_originals = Counter(r["original"] for r in fake_wrong if r["original"])
    correct_originals = Counter(r["original"] for r in fake_correct if r["original"])
    repeated_in_wrong = {k: v for k, v in wrong_originals.items() if v > 1}
    print(f"\nDistinct 'original' source videos behind missed fakes: {len(wrong_originals)} "
          f"(of {len(fake_wrong)} missed fakes) - repeats: {repeated_in_wrong}")

    print("=" * 60)


if __name__ == "__main__":
    main()
