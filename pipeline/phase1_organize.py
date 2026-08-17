from pathlib import Path
import re
import zipfile
import csv
import os
import json
import patoolib

RAW_DIR = Path("data/raw")
FAKE_AV_ZIP = RAW_DIR / "FakeAVCeleb_v1.2.zip"
FAKE_AV_DIR = RAW_DIR / "FakeAVCeleb_v1.2"

POLY_RAR = RAW_DIR / "PolyGlotFake.rar"
POLY_DIR = RAW_DIR / "PolyGlotFake"
POLY_FAKE_JSON_DIR = POLY_DIR / "PolyGlotFake" / "json_file" / "fake_Json_file"

FFPP_REAL_DIR = RAW_DIR / "FFPP" / "real"
FFPP_FAKE_DIR = RAW_DIR / "FFPP" / "fake"

DFDC_DIR = RAW_DIR / "DFDC"
DFDC_METADATA = DFDC_DIR / "metadata.json"

OUTPUT_MANIFEST = Path("outputs/manifests/inventory.csv")


_FFPP_PAIR_RE = re.compile(r'^\d{3}_\d{3}$')
_FFPP_SINGLE_RE = re.compile(r'^\d{3}$')
_CELEBDF_PAIR_RE = re.compile(r'^id\d{1,3}_id\d{1,3}_\d+$')
_CELEBDF_SINGLE_RE = re.compile(r'^id\d{1,3}_\d+$')
_DFD_RE = re.compile(r'^\d{2}_\d{2}__.+__[A-Za-z0-9]+$')


def classify_ffpp_source(video_path: Path) -> str:
    """The directory data/raw/FFPP/{real,fake}/ was found (2026-08-16, reviewer
    identity-split audit) to contain a mix of datasets under one blanket
    'FFPP' label, not pure FaceForensics++: filename-pattern census over the
    8,627 files found 1,000+1,000 genuine FF++ (XXX.mp4 / XXX_YYY.mp4 3-digit
    ids), 589+5,639 Celeb-DF-v2 (idN_NNNN.mp4 / idN_idM_NNNN.mp4 - matches
    Celeb-DF-v2's well-known 590-real/5639-fake composition almost exactly),
    1 stray Google/Jigsaw DFD-convention file (actor1_actor2__action__hash.mp4),
    and 398 files (300 five-digit-only + 49 four-digit + 49 '_fake'-suffixed)
    of undetermined origin - too small/inconsistent to match any known public
    dataset's convention confidently, so left as an explicitly-flagged
    'FFPP_unresolved' bucket rather than guessed. See
    DFDC_GENERALIZATION_INVESTIGATION.md's identity-split section for the
    full audit. Splitting the source label this way is required for
    identity-safe splitting (pipeline/phase4_split.py), since each of these
    sub-sources needs different identity-parsing logic."""
    stem = video_path.stem
    if _FFPP_PAIR_RE.match(stem) or _FFPP_SINGLE_RE.match(stem):
        return "FFPP"
    if _CELEBDF_PAIR_RE.match(stem) or _CELEBDF_SINGLE_RE.match(stem):
        return "CelebDF"
    if _DFD_RE.match(stem):
        return "DFD"
    return "FFPP_unresolved"


def load_polyglotfake_sync_tech_map():
    """Maps output video filename -> sync_tech (Wav2Lip / video_retalking) by
    reading all to_*.json files. Used as the manipulation_type for PolyGlotFake fakes."""
    mapping = {}
    if not POLY_FAKE_JSON_DIR.exists():
        return mapping
    for json_path in POLY_FAKE_JSON_DIR.glob("to_*.json"):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for entry in data.get("video", []):
                name = entry.get("name")
                sync_tech = entry.get("sync_tech")
                if name and sync_tech:
                    mapping[name] = sync_tech.lower()
        except Exception:
            continue
    return mapping

print("Phase 1: Organizing datasets...")

# Part 1: FakeAVCeleb Extractions
if FAKE_AV_ZIP.exists() and not FAKE_AV_DIR.exists():
    print(f"Extracting {FAKE_AV_ZIP}... (This may take a few minutes)")
    with zipfile.ZipFile(FAKE_AV_ZIP, 'r') as zip_ref:
        zip_ref.extractall(RAW_DIR)

# Part 2: PolyGlotFake Extractions
if POLY_RAR.exists() and not POLY_DIR.exists():
    print(f"Extracting {POLY_RAR}... (This may take a while)")
    POLY_DIR.mkdir(exist_ok=True)
    patoolib.extract_archive(str(POLY_RAR), outdir=str(POLY_DIR))

print("Reading metadata to generate unified inventory...")
OUTPUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)

poly_sync_tech_map = load_polyglotfake_sync_tech_map()

with open(OUTPUT_MANIFEST, 'w', newline='', encoding='utf-8') as f_out:
    writer = csv.writer(f_out)
    writer.writerow(["video_path", "label", "original_type", "race", "gender", "source", "manipulation_type"])
    count = 0

    # Process FakeAVCeleb
    meta_csv = FAKE_AV_DIR / "meta_data.csv"
    if meta_csv.exists():
        with open(meta_csv, 'r', encoding='utf-8') as f_in:
            reader = csv.reader(f_in)
            next(reader) # headers
            for row in reader:
                if len(row) < 10: continue
                method = row[3]
                orig_type = row[5]
                race = row[6]
                gender = row[7]
                filename = row[8]
                dir_path = row[9]

                label = 'real' if method == 'real' else 'fake'
                rel_dir = dir_path.replace("FakeAVCeleb/", "FakeAVCeleb_v1.2/", 1)
                full_path = str(RAW_DIR / rel_dir / filename)

                # manipulation_type: 'real' or the specific method (faceswap, fsgan, wav2lip, ...)
                manipulation_type = method

                writer.writerow([full_path.replace("\\", "/"), label, orig_type, race, gender, "FakeAVCeleb", manipulation_type])
                count += 1

    # Process PolyGlotFake
    if POLY_DIR.exists():
        for ext in ["*.mp4", "*.avi", "*.mov"]:
            for video_path in POLY_DIR.rglob(ext):
                path_str = str(video_path).replace("\\", "/")

                label = 'fake'
                if '/real/' in path_str or 'Celeb-real' in path_str or 'YouTube-real' in path_str:
                    label = 'real'
                elif '/fake/' in path_str or 'Celeb-synthesis' in path_str:
                    label = 'fake'
                elif 'List_of_testing_videos' in path_str:
                    continue # text file

                if label == 'real':
                    manipulation_type = 'real'
                else:
                    manipulation_type = poly_sync_tech_map.get(video_path.name, 'unknown_fake')

                # Polyglot doesn't have original_type/race/gender structured metadata
                writer.writerow([path_str, label, "", "", "", "PolyGlotFake", manipulation_type])
                count += 1

    # Process FFPP directory. Despite the directory name, this is NOT pure
    # FaceForensics++ - see classify_ffpp_source() above for the full audit.
    # Each file is reclassified to its actual source (FFPP / CelebDF / DFD /
    # FFPP_unresolved) by filename convention, since identity-safe splitting
    # (phase4_split.py) needs per-source identity-parsing logic and the paper
    # needs to name Celeb-DF as a training source rather than silently
    # folding it into "FaceForensics++".
    if FFPP_REAL_DIR.exists():
        for video_path in FFPP_REAL_DIR.glob("*.mp4"):
            path_str = str(video_path).replace("\\", "/")
            source = classify_ffpp_source(video_path)
            writer.writerow([path_str, "real", "", "", "", source, "real"])
            count += 1

    if FFPP_FAKE_DIR.exists():
        for video_path in FFPP_FAKE_DIR.glob("*.mp4"):
            path_str = str(video_path).replace("\\", "/")
            source = classify_ffpp_source(video_path)
            writer.writerow([path_str, "fake", "", "", "", source, "unknown_fake"])
            count += 1

    # Process DFDC (partial). Reserved entirely as a held-out cross-dataset
    # test set - pipeline/phase4_split.py routes source=="DFDC" rows straight
    # to held_out_crossdataset.csv, never into train/val/test.
    if DFDC_METADATA.exists():
        with open(DFDC_METADATA, 'r', encoding='utf-8') as f:
            dfdc_meta = json.load(f)
        for filename, info in dfdc_meta.items():
            video_path = DFDC_DIR / filename
            if not video_path.exists():
                continue
            label = "real" if info.get("label") == "REAL" else "fake"
            manipulation_type = "real" if label == "real" else "unknown_fake"
            path_str = str(video_path).replace("\\", "/")
            writer.writerow([path_str, label, "", "", "", "DFDC", manipulation_type])
            count += 1

print(f"Inventory built! Organized {count} items across multiple datasets.")
