"""
Dataset over pre-cached DINOv2 + Whisper features (training/extract_features.py
output). Used for Stage A of staged training (frozen backbone).

Each .pt file already contains: visual [8,4,384], audio [8,384], metadata [34],
label (float), manipulation_type (str) - no image/audio decoding at train time.
"""
import csv
import hashlib
import random
from pathlib import Path
import torch
from torch.utils.data import Dataset

from model import MANIPULATION_TYPE_TO_IDX, DOMAIN_TO_IDX


def _make_video_id(video_path: str) -> str:
    """Must match training/extract_features.py's make_video_id exactly."""
    p = Path(video_path)
    try:
        rel = p.relative_to(Path("data/raw"))
    except ValueError:
        rel = p
    return f"{p.stem}_{hashlib.md5(str(rel).replace(chr(92), '/').encode('utf-8')).hexdigest()[:10]}"


def _load_source_lookup(split_name: str) -> dict:
    """video_id -> dataset source ('FakeAVCeleb'/'PolyGlotFake'/'FFPP'), built
    from the manifest CSV for the domain-adversarial branch. Cached .pt files
    from before this feature was added don't carry 'source' directly (see
    training/extract_features.py's torch.save dict) - reconstructing it here
    avoids needing to re-run the ~2.5-3hr feature-extraction job. Any future
    re-extraction will save 'source' directly and this lookup becomes a
    (harmless) fallback - see __getitem__ below."""
    manifest_name = "held_out_crossdataset" if split_name == "held_out" else split_name
    manifest_path = Path(f"outputs/manifests/{manifest_name}.csv")
    lookup = {}
    if not manifest_path.exists():
        return lookup
    with open(manifest_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            lookup[_make_video_id(row["video_path"])] = row.get("source", "unknown")
    return lookup


class CachedFeatureDataset(Dataset):
    def __init__(self, root_dir: str, is_train: bool = False):
        self.root_dir = Path(root_dir)
        self.is_train = is_train
        self.items = []  # list of (path, label)
        self._source_lookup = _load_source_lookup(self.root_dir.name)
        self._load()

    def _load(self):
        real_items, fake_items = [], []
        for label_name, label_val in [("real", 0.0), ("fake", 1.0)]:
            class_dir = self.root_dir / label_name
            if not class_dir.exists():
                continue
            for pt_file in sorted(class_dir.glob("*.pt")):
                (real_items if label_val == 0.0 else fake_items).append((pt_file, label_val))

        if self.is_train and real_items:
            target_fake_count = 3 * len(real_items)
            if len(fake_items) > target_fake_count:
                random.seed(42)
                fake_items = random.sample(fake_items, target_fake_count)
            combined = real_items + fake_items
            random.seed(42)
            random.shuffle(combined)
            self.items = combined
        else:
            self.items = real_items + fake_items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        data = torch.load(path, map_location="cpu", weights_only=False)
        manip_type = data.get("manipulation_type", "unknown_fake")
        manip_idx = MANIPULATION_TYPE_TO_IDX.get(manip_type, MANIPULATION_TYPE_TO_IDX["unknown_fake"])
        # Prefer "source" directly on the cached tensor (present for any cache
        # regenerated after extract_features.py started saving it); fall back
        # to the manifest-based lookup for the existing cache, which predates it.
        source = data.get("source") or self._source_lookup.get(path.stem, "unknown")
        domain_idx = DOMAIN_TO_IDX.get(source, -100)  # -100 = CrossEntropyLoss's default ignore_index
        return (
            data["visual"],
            data["audio"],
            data["metadata"],
            torch.tensor(label, dtype=torch.float32),
            torch.tensor(manip_idx, dtype=torch.long),
            torch.tensor(domain_idx, dtype=torch.long),
        )
