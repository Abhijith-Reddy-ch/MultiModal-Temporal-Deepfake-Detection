"""
Phase D - Feature Caching with Frozen SSL Backbones (plan.pdf Phase 3).

Runs DINOv2 ViT-S/14 (visual) and Whisper-tiny encoder (audio) ONCE, frozen,
over every video's cached crops (from pipeline/phase5_extract.py) and caches
compact tensors to disk. Training never re-runs these backbones live.

Videos are processed in batches (VIDEO_BATCH at a time) so both backbones run
one larger GPU forward pass per batch instead of one tiny pass per video -
whisper-tiny's encoder always operates on a fixed 1500-token (30s-padded)
sequence per its architecture (a strict shape assertion inside the library
rules out truncating that), so batching across videos is the safe way to cut
per-video Python/CUDA-launch overhead without touching library internals.

Output per video: data/processed/features_cached/{split}/{label}/{video_id}.pt
  {
    "visual": FloatTensor [8, 4, 384]   # frames x regions x DINOv2 CLS dim
    "audio":  FloatTensor [8, 384]      # frames x Whisper encoder dim
    "metadata": FloatTensor [34]        # ffprobe-derived metadata vector
    "label": float (0.0=real, 1.0=fake)
    "manipulation_type": str
  }
"""
import os
import sys
import csv
import json
import hashlib
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from torchvision import transforms

sys.path.append(os.getcwd())

import whisper

CROPS_DIR = Path("data/processed/crops")
CACHE_DIR = Path("data/processed/features_cached")
FRAMES_PER_VIDEO = 8
REGIONS = ["face", "eyes", "lips", "jaw"]
DINO_DIM = 384
VIDEO_BATCH = 24  # videos per GPU batch - fp16 autocast roughly halves activation memory, so we can go bigger

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]

_dino_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
])


# ------------------------------------------------------------------ #
#  Metadata vector (34-dim, ffprobe-derived) - same scheme as the     #
#  legacy training/dataset.py encoder, kept for continuity.           #
# ------------------------------------------------------------------ #
def encode_metadata(meta: dict) -> torch.Tensor:
    file_size = float(meta.get("file_size", 0.0) or 0.0)
    log_file_size = np.log10(file_size + 1.0) / 10.0
    duration = float(meta.get("duration", 0.0) or 0.0)
    norm_duration = min(duration / 60.0, 1.0)
    nb_streams = float(meta.get("nb_streams", 0.0) or 0.0)
    norm_nb_streams = min(nb_streams / 5.0, 1.0)
    width = float(meta.get("width", 0.0) or 0.0)
    norm_width = width / 1920.0
    height = float(meta.get("height", 0.0) or 0.0)
    norm_height = height / 1080.0
    fps = float(meta.get("fps", 0.0) or 0.0)
    norm_fps = fps / 60.0
    video_bitrate = float(meta.get("video_bitrate", 0.0) or 0.0)
    log_v_bitrate = np.log10(video_bitrate + 1.0) / 10.0
    audio_bitrate = float(meta.get("audio_bitrate", 0.0) or 0.0)
    log_a_bitrate = np.log10(audio_bitrate + 1.0) / 10.0
    audio_sample_rate = float(meta.get("audio_sample_rate", 0.0) or 0.0)
    norm_sample_rate = audio_sample_rate / 48000.0
    audio_channels = float(meta.get("audio_channels", 0.0) or 0.0)
    norm_channels = audio_channels / 6.0
    gop_length = float(meta.get("gop_length", 0.0) or 0.0)
    norm_gop = min(gop_length / 250.0, 1.0)
    nb_frames = float(meta.get("nb_frames", 0.0) or 0.0)
    norm_nb_frames = min(nb_frames / 1800.0, 1.0)
    rotation = float(meta.get("rotation", 0.0) or 0.0)
    norm_rotation = rotation / 360.0

    numeric_feats = [
        log_file_size, norm_duration, norm_nb_streams, norm_width, norm_height,
        norm_fps, log_v_bitrate, log_a_bitrate, norm_sample_rate, norm_channels,
        norm_gop, norm_nb_frames, norm_rotation,
    ]

    video_codec = meta.get("video_codec", "unknown")
    v_codec_onehot = [0.0] * 5
    v_codec_onehot[{"h264": 0, "hevc": 1, "vp9": 2, "mpeg4": 3}.get(video_codec, 4)] = 1.0

    audio_codec = meta.get("audio_codec", "unknown")
    a_codec_onehot = [0.0] * 5
    a_codec_onehot[{"aac": 0, "mp3": 1, "pcm_s16le": 2, "opus": 3}.get(audio_codec, 4)] = 1.0

    pix_fmt = meta.get("pix_fmt", "unknown")
    pix_onehot = [0.0] * 3
    pix_onehot[{"yuv420p": 0, "yuvj420p": 1}.get(pix_fmt, 2)] = 1.0

    container = meta.get("container_format", "unknown")
    c_idx = 0 if ("mp4" in container or "mov" in container or "m4a" in container) else (1 if "avi" in container else 2)
    c_onehot = [0.0] * 3
    c_onehot[c_idx] = 1.0

    bool_feats = [
        1.0 if meta.get("vfr", False) else 0.0,
        1.0 if meta.get("has_creation_time", False) else 0.0,
        1.0 if meta.get("missing_audio", True) else 0.0,
        1.0 if meta.get("missing_video", True) else 0.0,
        1.0 if meta.get("has_tags", False) else 0.0,
    ]

    feats = numeric_feats + v_codec_onehot + a_codec_onehot + pix_onehot + c_onehot + bool_feats
    return torch.tensor(feats, dtype=torch.float32)


def make_video_id(video_path: str) -> str:
    p = Path(video_path)
    try:
        rel = p.relative_to(Path("data/raw"))
    except ValueError:
        rel = p
    return f"{p.stem}_{hashlib.md5(str(rel).replace(chr(92), '/').encode('utf-8')).hexdigest()[:10]}"


def load_video_dino_batch(video_dir: Path):
    """Loads all 8*4=32 region-frame crops for a video, returns a [32,3,224,224] tensor."""
    tensors = []
    for i in range(FRAMES_PER_VIDEO):
        for region in REGIONS:
            fp = video_dir / f"frame{i}_{region}.jpg"
            try:
                img = Image.open(fp).convert("RGB")
            except Exception:
                img = Image.new("RGB", (224, 224))
            tensors.append(_dino_transform(img))
    return torch.stack(tensors)  # [32, 3, 224, 224]


def load_mel(audio_path: Path, n_mels: int):
    """Loads audio and returns a fixed-size [n_mels, 3000] mel spectrogram (30s-padded,
    per whisper's fixed positional embedding table - can't be shortened safely)."""
    try:
        audio = whisper.load_audio(str(audio_path))
        audio = whisper.pad_or_trim(audio)
        return whisper.log_mel_spectrogram(audio, n_mels=n_mels)
    except Exception:
        return torch.zeros(n_mels, 3000)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading DINOv2 ViT-S/14 (frozen)...")
    dinov2 = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    dinov2.eval().to(device)
    for p in dinov2.parameters():
        p.requires_grad = False

    print("Loading Whisper-tiny encoder (frozen)...")
    whisper_model = whisper.load_model("tiny", device=device)
    whisper_model.eval()
    for p in whisper_model.parameters():
        p.requires_grad = False
    n_mels = whisper_model.dims.n_mels

    media_metadata = {}
    meta_json = Path("outputs/manifests/media_metadata.json")
    if meta_json.exists():
        with open(meta_json, "r", encoding="utf-8") as f:
            media_metadata = json.load(f)

    splits = {
        "train": Path("outputs/manifests/train.csv"),
        "val": Path("outputs/manifests/val.csv"),
        "test": Path("outputs/manifests/test.csv"),
        "held_out": Path("outputs/manifests/held_out_crossdataset.csv"),
    }

    total_processed = 0
    total_skipped = 0

    for split_name, csv_path in splits.items():
        if not csv_path.exists():
            continue
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        if not rows:
            print(f"[{split_name}] no rows, skipping.")
            continue

        print(f"[{split_name}] caching features for {len(rows)} videos...")

        # Build the list of videos that still need processing (skip already-cached / missing crops)
        pending = []
        for row in rows:
            video_path = row["video_path"]
            label_str = row["label"].lower()
            video_id = make_video_id(video_path)
            crop_dir = CROPS_DIR / split_name / label_str / video_id
            out_dir = CACHE_DIR / split_name / label_str
            out_path = out_dir / f"{video_id}.pt"

            if out_path.exists():
                total_skipped += 1
                continue
            if not crop_dir.exists():
                continue
            pending.append((row, video_id, label_str, crop_dir, out_dir, out_path))

        done_count = 0
        num_workers = min(16, os.cpu_count() or 1)
        with ThreadPoolExecutor(max_workers=num_workers) as pool:
            for batch_start in range(0, len(pending), VIDEO_BATCH):
                batch = pending[batch_start: batch_start + VIDEO_BATCH]
                if not batch:
                    continue

                # Disk I/O (32 JPEGs + 1 wav per video) is the real bottleneck once
                # the GPU forward pass is fast (fp16) - parallelize file reads across
                # threads (PIL/soundfile release the GIL during decode) instead of
                # loading one video's files at a time.
                frame_batches = list(pool.map(load_video_dino_batch, [item[3] for item in batch]))
                mel_batches = list(pool.map(lambda item: load_mel(item[3] / "audio.wav", n_mels), batch))

                all_frames = torch.cat(frame_batches, dim=0).to(device)  # [B*32, 3, 224, 224]
                all_mels = torch.stack(mel_batches, dim=0).to(device)     # [B, n_mels, 3000]

                with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                    cls_tokens = dinov2(all_frames)  # [B*32, 384]
                    hidden = whisper_model.encoder(all_mels)  # [B, 1500, 384]

                B = len(batch)
                cls_tokens = cls_tokens.view(B, FRAMES_PER_VIDEO, len(REGIONS), DINO_DIM).float().cpu()

                hidden = hidden.transpose(1, 2)  # [B, 384, 1500]
                pooled = F.adaptive_avg_pool1d(hidden.float(), FRAMES_PER_VIDEO)  # [B, 384, 8]
                pooled = pooled.transpose(1, 2).cpu()  # [B, 8, 384]

                for i, (row, video_id, label_str, crop_dir, out_dir, out_path) in enumerate(batch):
                    video_path = row["video_path"]
                    label = 0.0 if label_str == "real" else 1.0
                    manipulation_type = row.get("manipulation_type", "unknown")

                    meta = media_metadata.get(video_path, {})
                    metadata_vec = encode_metadata(meta)

                    out_dir.mkdir(parents=True, exist_ok=True)
                    torch.save({
                        "visual": cls_tokens[i],
                        "audio": pooled[i],
                        "metadata": metadata_vec,
                        "label": label,
                        "manipulation_type": manipulation_type,
                        "source": row.get("source", "unknown"),
                    }, out_path)

                done_count += len(batch)
                total_processed += len(batch)
                if done_count % 500 < VIDEO_BATCH:
                    print(f"  [{split_name}] {done_count}/{len(pending)} done...")

    print(f"Feature caching complete! Processed {total_processed} new, skipped {total_skipped} already-cached.")


if __name__ == "__main__":
    main()
