"""
Dataset over raw crops/audio (pipeline/phase5_extract.py output) - used for
Stage B, where DINOv2/Whisper are no longer frozen (LoRA adapters need live
gradients through them), so features can't be pre-cached.
"""
import io
import random
import hashlib
from pathlib import Path
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageFilter
from torchvision import transforms

import whisper
from model import MANIPULATION_TYPE_TO_IDX, DOMAIN_TO_IDX

CROPS_DIR = Path("data/processed/crops")
FRAMES_PER_VIDEO = 8
REGIONS = ["face", "eyes", "lips", "jaw"]

SBI_MANIFEST = Path("outputs/manifests/train_sbi.csv")
SBI_TARGET_MULTIPLIER = 1.0  # cap SBI pseudo-fakes at this multiple of real count,
                             # added on top of the existing 3x-real fake_items cap

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
])
_normalize_to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
])


def sample_video_augmentation():
    """Samples ONE set of augmentation params per video, applied identically
    across all its frames/regions - augmentation should reflect a property of
    the source clip's pipeline (compression, resolution, flip), not vary
    frame-to-frame, or it would break the temporal consistency the
    LSTM/fusion transformer is supposed to learn from."""
    return {
        # JPEG re-encode is the expensive op (full codec round-trip via PIL) -
        # cut its probability to reduce CPU cost; keep the cheap ops (bilinear
        # resize, blur, jitter) at similar rates since they don't bottleneck.
        "jpeg_quality": random.randint(25, 90) if random.random() < 0.2 else None,
        "res_scale": random.uniform(0.4, 0.9) if random.random() < 0.4 else None,
        "blur_radius": random.uniform(0.3, 2.0) if random.random() < 0.3 else None,
        "flip": random.random() < 0.5,
        "brightness": random.uniform(0.8, 1.2),
        "contrast": random.uniform(0.8, 1.2),
        "saturation": random.uniform(0.8, 1.2),
    }


def apply_video_augmentation(img: Image.Image, params: dict, size: int = 224) -> Image.Image:
    img = img.resize((size, size), Image.BILINEAR)

    if params["res_scale"] is not None:
        small = max(8, int(size * params["res_scale"]))
        img = img.resize((small, small), Image.BILINEAR).resize((size, size), Image.BILINEAR)

    if params["jpeg_quality"] is not None:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=params["jpeg_quality"])
        buf.seek(0)
        img = Image.open(buf).convert("RGB")

    if params["blur_radius"] is not None:
        img = img.filter(ImageFilter.GaussianBlur(radius=params["blur_radius"]))

    if params["flip"]:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)

    img = transforms.functional.adjust_brightness(img, params["brightness"])
    img = transforms.functional.adjust_contrast(img, params["contrast"])
    img = transforms.functional.adjust_saturation(img, params["saturation"])

    return img


def make_video_id(video_path: str) -> str:
    p = Path(video_path)
    try:
        rel = p.relative_to(Path("data/raw"))
    except ValueError:
        rel = p
    return f"{p.stem}_{hashlib.md5(str(rel).replace(chr(92), '/').encode('utf-8')).hexdigest()[:10]}"


def make_sbi_id(video_path: str) -> str:
    """Must match pipeline/phase5c_sbi_augment.py's make_sbi_id exactly."""
    return make_video_id(f"{video_path}::sbi")


class RawCropsDataset(Dataset):
    """Returns (images [32,3,224,224], mel [n_mels,3000], metadata [34], label,
    manip_idx, domain_idx). domain_idx is -100 (CrossEntropyLoss's ignore_index)
    for rows whose 'source' isn't one of the 3 training-domain names (e.g. SBI
    pseudo-fakes, or the held-out DFDC split, which never gets a domain id)."""

    def __init__(self, split_name: str, n_mels: int = 80, is_train: bool = False, use_sbi: bool = False,
                 use_augmentation: bool = True):
        import csv
        import json

        self.split_name = split_name
        self.n_mels = n_mels
        self.is_train = is_train
        self.use_augmentation = use_augmentation

        media_metadata = {}
        meta_json = Path("outputs/manifests/media_metadata.json")
        if meta_json.exists():
            with open(meta_json, "r", encoding="utf-8") as f:
                media_metadata = json.load(f)
        self.media_metadata = media_metadata

        # held_out's manifest is named held_out_crossdataset.csv (see pipeline/phase4_split.py),
        # but crop/feature directories still use the plain "held_out" split name.
        manifest_name = "held_out_crossdataset" if split_name == "held_out" else split_name
        csv_path = Path(f"outputs/manifests/{manifest_name}.csv")
        real_items, fake_items = [], []
        if csv_path.exists():
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    label_str = row["label"].lower()
                    video_id = make_video_id(row["video_path"])
                    crop_dir = CROPS_DIR / split_name / label_str / video_id
                    if not crop_dir.exists():
                        continue
                    item = (crop_dir, row)
                    (real_items if label_str == "real" else fake_items).append(item)

        if is_train and real_items:
            target_fake = 3 * len(real_items)
            if len(fake_items) > target_fake:
                random.seed(42)
                fake_items = random.sample(fake_items, target_fake)

            # Self-Blended Images (pipeline/phase5c_sbi_augment.py) - added ON TOP
            # of the 3x-real fake cap above, not counted against it, so real fake
            # diversity isn't crowded out by synthetic pseudo-fakes.
            if use_sbi and split_name == "train" and SBI_MANIFEST.exists():
                sbi_items = []
                with open(SBI_MANIFEST, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        orig_video_path = row["video_path"].rsplit("::sbi", 1)[0]
                        sbi_id = make_sbi_id(orig_video_path)
                        crop_dir = CROPS_DIR / "train" / "fake_sbi" / sbi_id
                        if crop_dir.exists():
                            sbi_items.append((crop_dir, row))

                target_sbi = int(SBI_TARGET_MULTIPLIER * len(real_items))
                if len(sbi_items) > target_sbi:
                    random.seed(42)
                    sbi_items = random.sample(sbi_items, target_sbi)
                fake_items = fake_items + sbi_items

            combined = real_items + fake_items
            random.seed(42)
            random.shuffle(combined)
            self.items = combined
        else:
            self.items = real_items + fake_items

    def __len__(self):
        return len(self.items)

    def _load_images(self, crop_dir: Path):
        aug_params = sample_video_augmentation() if (self.is_train and self.use_augmentation) else None
        tensors = []
        for i in range(FRAMES_PER_VIDEO):
            for region in REGIONS:
                fp = crop_dir / f"frame{i}_{region}.jpg"
                try:
                    img = Image.open(fp).convert("RGB")
                except Exception:
                    img = Image.new("RGB", (224, 224))
                if aug_params is not None:
                    img = apply_video_augmentation(img, aug_params)
                    tensors.append(_normalize_to_tensor(img))
                else:
                    tensors.append(_transform(img))
        return torch.stack(tensors)  # [32, 3, 224, 224]

    def _load_mel(self, crop_dir: Path):
        try:
            audio = whisper.load_audio(str(crop_dir / "audio.wav"))
            audio = whisper.pad_or_trim(audio)
            return whisper.log_mel_spectrogram(audio, n_mels=self.n_mels)
        except Exception:
            return torch.zeros(self.n_mels, 3000)

    def encode_metadata(self, video_path: str) -> torch.Tensor:
        import numpy as np
        meta = self.media_metadata.get(video_path, {})

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

    def __getitem__(self, idx):
        crop_dir, row = self.items[idx]
        label_str = row["label"].lower()
        label = 0.0 if label_str == "real" else 1.0
        manip_type = row.get("manipulation_type", "unknown_fake")
        manip_idx = MANIPULATION_TYPE_TO_IDX.get(manip_type, MANIPULATION_TYPE_TO_IDX["unknown_fake"])
        domain_idx = DOMAIN_TO_IDX.get(row.get("source", "unknown"), -100)

        images = self._load_images(crop_dir)
        mel = self._load_mel(crop_dir)
        metadata = self.encode_metadata(row["video_path"])

        return (
            images,
            mel,
            metadata,
            torch.tensor(label, dtype=torch.float32),
            torch.tensor(manip_idx, dtype=torch.long),
            torch.tensor(domain_idx, dtype=torch.long),
        )
