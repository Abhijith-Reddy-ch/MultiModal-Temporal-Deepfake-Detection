"""One-off: evaluate the Stage B model on just the DFDC held-out set, reusing
the threshold already determined from the full evaluate_stage_b.py run (0.10),
to avoid re-running the slow val+test passes again."""
import os
import sys
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'training'))

import numpy as np
import torch
from torch.utils.data import DataLoader

from raw_crops_dataset import RawCropsDataset
from evaluate_stage_b import load_stage_b_models, get_probs_labels, report_table

THRESHOLD = 0.10  # from the full evaluate_stage_b.py run's validation-based selection

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dinov2, whisper_encoder, classifier, n_mels = load_stage_b_models(device)

    held_out_dataset = RawCropsDataset("held_out", n_mels=n_mels, is_train=False)
    print(f"held_out dataset size: {len(held_out_dataset)}")

    held_out_loader = DataLoader(held_out_dataset, batch_size=4, shuffle=False, num_workers=min(4, os.cpu_count() or 1))
    ho_labels, ho_probs = get_probs_labels(dinov2, whisper_encoder, classifier, held_out_loader, device)
    report_table("Cross-Dataset Generalization (DFDC held-out) - Stage B", ho_labels, ho_probs, THRESHOLD)


if __name__ == "__main__":
    main()
