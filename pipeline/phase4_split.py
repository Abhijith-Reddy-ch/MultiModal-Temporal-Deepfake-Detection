import csv
import random
import re
import hashlib
from pathlib import Path
from collections import defaultdict

# Phase 3 (video normalization) has been dropped - frame/region extraction now
# reads directly from data/raw/, so splitting works off the cleaned inventory.
MANIFEST_FILE = Path("outputs/manifests/inventory.csv")
TRAIN_FILE = Path("outputs/manifests/train.csv")
VAL_FILE = Path("outputs/manifests/val.csv")
TEST_FILE = Path("outputs/manifests/test.csv")

# Reserved for DFDC (or any other fully held-out cross-dataset test set). Empty
# until that data is added - never populated from FakeAVCeleb/PolyGlotFake/FF++,
# and never touched by training or threshold selection.
HELD_OUT_FILE = Path("outputs/manifests/held_out_crossdataset.csv")
HELD_OUT_SOURCES = {"DFDC"}

_FFPP_STEM_RE = re.compile(r'(\d{3})(?:_(\d{3}))?$')
_CELEBDF_STEM_RE = re.compile(r'(id\d{1,3})(?:_(id\d{1,3}))?_\d+$')
_DFD_STEM_RE = re.compile(r'(\d{2})_(\d{2})__')


def extract_identities(path_str, source=None):
    """Identity-safe splitting needs real identity groups, not just
    'something unique per video'. Per-source parsing (added 2026-08-16, see
    phase1_organize.py::classify_ffpp_source for how 'source' values were
    corrected):
      - FFPP (genuine FaceForensics++ only): filename is the 3-digit source
        video id ('000.mp4') or a manipulated pair ('000_003.mp4' = swap
        between source videos 000 and 003) - both numeric ids are identities.
      - CelebDF (Celeb-DF-v2): filename is 'idN_NNNN.mp4' (real) or
        'idN_idM_NNNN.mp4' (synthesis) - the idN/idM tokens are identities.
      - DFD: filename is 'AA_BB__action__hash.mp4' - AA/BB are the two
        actor ids involved.
      - Everything else (FakeAVCeleb, PolyGlotFake, FFPP_unresolved, DFD
        videos that don't match the pattern): fall back to the original
        id\\d{5} regex (FakeAVCeleb's convention), then to a per-video MD5
        hash if that finds nothing - i.e. no real identity grouping, each
        video is its own singleton identity. This is an honest "we don't
        know this video's identity" rather than a silent false claim of
        identity-disjointness.
    """
    stem = Path(path_str).stem

    if source == "FFPP":
        m = _FFPP_STEM_RE.search(stem)
        if m:
            return {f"ffpp_{g}" for g in m.groups() if g}

    if source == "CelebDF":
        m = _CELEBDF_STEM_RE.search(stem)
        if m:
            return {f"celebdf_{g}" for g in m.groups() if g}

    if source == "DFD":
        m = _DFD_STEM_RE.search(stem)
        if m:
            return {f"dfd_{g}" for g in m.groups() if g}

    # Fallback: FakeAVCeleb-style 'idXXXXX' tags, else a per-video MD5 hash
    # (FFPP_unresolved and any source-specific pattern that failed to match
    # land here - no real identity info available for these).
    ids = set(re.findall(r'id\d{5}', path_str))
    if not ids:
        unique_hash = "hash_" + hashlib.md5(path_str.encode('utf-8')).hexdigest()[:8]
        ids.add(unique_hash)
    return ids

class _UnionFind:
    """Minimal union-find over identity tokens. Needed because a video can
    carry MULTIPLE identities (e.g. FakeAVCeleb/FF++/CelebDF face-swap
    videos name both the source-face donor and the target-video identity) -
    assigning identity tokens to splits independently (the pre-2026-08-16
    approach) does not guarantee disjointness: if identity A's own video
    goes to train but A is also paired with identity B (assigned to test)
    in a swap video, that swap video must go to test, and now A has
    representation in both train and test. The fix is to union every pair
    of identities that ever co-occur in the same video, then split at the
    connected-component level, so no identity can ever end up split across
    two partitions."""

    def __init__(self):
        self.parent = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def main():
    print("Phase 4: Splitting dataset (identity-disjoint 80/10/10, connected-component-safe)...")

    if not MANIFEST_FILE.exists():
        raise FileNotFoundError("Inventory file not found! Did you run phase 1/2?")

    random.seed(42) # For reproducibility

    videos = []
    held_out_rows = []

    with open(MANIFEST_FILE, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        headers = next(reader)
        source_idx = headers.index("source")

        for row in reader:
            if row[source_idx] in HELD_OUT_SOURCES:
                held_out_rows.append(row)
                continue
            video_path = row[0]
            ids = extract_identities(video_path, source=row[source_idx])
            videos.append((row, ids))

    print(f"Found {len(videos)} trainable videos.")
    print(f"Held out as cross-dataset test (untouched by train/val/threshold): {len(held_out_rows)} videos.")

    # Union every pair of identities that co-occur in the same video (see
    # _UnionFind docstring), then group videos by connected component.
    uf = _UnionFind()
    for _, ids in videos:
        ids = list(ids)
        for i in range(1, len(ids)):
            uf.union(ids[0], ids[i])
        if ids:
            uf.find(ids[0])  # register singleton components too

    component_videos = defaultdict(list)
    for row, ids in videos:
        if not ids:
            continue
        root = uf.find(next(iter(ids)))
        component_videos[root].append(row)

    total_videos = sum(len(v) for v in component_videos.values())
    print(f"Collapsed into {len(component_videos)} identity-connected components "
          f"(components can span multiple raw identity tokens when videos link them).")

    # Components vary hugely in size (a few giant components from heavily
    # cross-paired face-swap identities can cover >10% of the dataset each -
    # see DFDC_GENERALIZATION_INVESTIGATION.md's identity-split audit), so a
    # naive "shuffle components, take the first 80%" would badly skew video
    # counts. Balanced greedy bin-packing (largest components first, always
    # placed into whichever split is furthest below its target share) keeps
    # video-count proportions close to 80/10/10 despite the skew.
    components = list(component_videos.items())
    random.shuffle(components)  # reproducible tie-break among equal-size components
    components.sort(key=lambda kv: len(kv[1]), reverse=True)

    targets = {"train": 0.80 * total_videos, "val": 0.10 * total_videos, "test": 0.10 * total_videos}
    current = {"train": 0, "val": 0, "test": 0}
    bucket_rows = {"train": [], "val": [], "test": []}

    for _, rows in components:
        split = max(current, key=lambda s: targets[s] - current[s])
        bucket_rows[split].extend(rows)
        current[split] += len(rows)

    train_rows, val_rows, test_rows = bucket_rows["train"], bucket_rows["val"], bucket_rows["test"]

    # Write splits
    for split_file, rows, name in [(TRAIN_FILE, train_rows, "Train"),
                                  (VAL_FILE, val_rows, "Val"),
                                  (TEST_FILE, test_rows, "Test")]:
        with open(split_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(rows)
            pct = (len(rows) / len(videos) * 100) if videos else 0.0
            print(f"  {name}: {len(rows)} videos ({pct:.1f}%) saved to {split_file}")

    with open(HELD_OUT_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(held_out_rows)
        print(f"  Held-out cross-dataset: {len(held_out_rows)} videos saved to {HELD_OUT_FILE}")

    print("Phase 4 completed successfully!")

if __name__ == "__main__":
    main()
