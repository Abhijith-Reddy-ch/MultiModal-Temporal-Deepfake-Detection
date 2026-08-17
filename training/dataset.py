import os
import hashlib
import numpy as np
from pathlib import Path
from PIL import Image
from collections import defaultdict
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms

RAW_DIR = Path("data/raw")


def make_video_id(orig_path: str) -> str:
    """Must match the unique id scheme used by pipeline/phase5_extract.py."""
    p = Path(orig_path)
    try:
        rel = p.relative_to(RAW_DIR)
    except ValueError:
        rel = p
    unique_hash = hashlib.md5(str(rel).replace("\\", "/").encode("utf-8")).hexdigest()[:10]
    return f"{p.stem}_{unique_hash}"

# ------------------------------------------------------------------ #
#  Lazy imports for optional heavy libraries (mediapipe, librosa)     #
# ------------------------------------------------------------------ #
try:
    import mediapipe as mp
    _MP_AVAILABLE = True
except ImportError:
    _MP_AVAILABLE = False

try:
    import librosa
    _LIBROSA_AVAILABLE = True
except ImportError:
    _LIBROSA_AVAILABLE = False

# ------------------------------------------------------------------ #
#  Constants                                                           #
# ------------------------------------------------------------------ #
FRAMES_PER_VIDEO = 5        # How many frames we sample per video
IMG_SIZE = 224              # Input resolution for each branch
MFCC_BINS = 40              # Number of MFCC coefficients
MFCC_TIME = 128             # Fixed time-length for MFCC tensor

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]


# ------------------------------------------------------------------ #
#  Face-region helper                                                  #
# ------------------------------------------------------------------ #
class FaceRegionExtractor:
    """
    Uses MediaPipe Face Landmarker (v0.10+ API) to detect the face and
    crop three regions:
        - full face  (224x224)
        - lip region (lower 40% of face bbox)
        - eye region (upper 40% of face bbox)
    Falls back to the full image if detection fails or mediapipe is unavailable.

    Note: The landmarker is created lazily on first use so that this class
    is picklable (safe for DataLoader multiprocessing workers).
    """

    def __init__(self):
        # Don't create the landmarker here — it's not picklable.
        # It will be created on first call to extract() within each worker process.
        self._landmarker = None
        self._init_attempted = False

    def _init_landmarker(self):
        """Create the mediapipe FaceLandmarker (called once per worker process)."""
        if self._init_attempted:
            return
        self._init_attempted = True
        if not _MP_AVAILABLE:
            return
        try:
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision
            import urllib.request, tempfile, os as _os

            model_url = (
                "https://storage.googleapis.com/mediapipe-models/"
                "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
            )
            cache_path = _os.path.join(tempfile.gettempdir(), "face_landmarker.task")
            if not _os.path.exists(cache_path):
                urllib.request.urlretrieve(model_url, cache_path)

            base_opts = mp_python.BaseOptions(model_asset_path=cache_path)
            opts = mp_vision.FaceLandmarkerOptions(
                base_options=base_opts,
                num_faces=1,
                min_face_detection_confidence=0.4,
            )
            self._landmarker = mp_vision.FaceLandmarker.create_from_options(opts)
        except Exception:
            self._landmarker = None

    def extract(self, pil_img):
        """Returns (face_pil, lip_pil, eye_pil) all resized to IMG_SIZE."""
        self._init_landmarker()     # no-op after first call

        w, h = pil_img.size

        if self._landmarker is not None:
            try:
                import mediapipe as mp
                img_rgb = np.array(pil_img)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
                result = self._landmarker.detect(mp_image)
                if result.face_landmarks:
                    lm = result.face_landmarks[0]
                    xs = [p.x * w for p in lm]
                    ys = [p.y * h for p in lm]
                    x1, x2 = max(0, int(min(xs))), min(w, int(max(xs)))
                    y1, y2 = max(0, int(min(ys))), min(h, int(max(ys)))
                    fh = max(1, y2 - y1)
                    face_crop = pil_img.crop((x1, y1, x2, y2))
                    eye_crop  = pil_img.crop((x1, y1, x2, y1 + int(fh * 0.4)))
                    lip_crop  = pil_img.crop((x1, y1 + int(fh * 0.6), x2, y2))
                    return (
                        face_crop.resize((IMG_SIZE, IMG_SIZE)),
                        lip_crop.resize((IMG_SIZE, IMG_SIZE)),
                        eye_crop.resize((IMG_SIZE, IMG_SIZE)),
                    )
            except Exception:
                pass

        # Fallback: use full frame for all branches
        resized = pil_img.resize((IMG_SIZE, IMG_SIZE))
        return resized, resized, resized



# ------------------------------------------------------------------ #
#  MFCC audio helper                                                   #
# ------------------------------------------------------------------ #
def load_mfcc(audio_path: Path, n_mfcc: int = MFCC_BINS, target_len: int = MFCC_TIME):
    """Load a .wav and return a (1, n_mfcc, target_len) float32 tensor."""
    if _LIBROSA_AVAILABLE and audio_path.exists():
        try:
            y, sr = librosa.load(str(audio_path), sr=16000, mono=True)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)  # (n_mfcc, T)
            # Normalise
            mfcc = (mfcc - mfcc.mean()) / (mfcc.std() + 1e-6)
            # Pad or trim
            if mfcc.shape[1] >= target_len:
                mfcc = mfcc[:, :target_len]
            else:
                pad = target_len - mfcc.shape[1]
                mfcc = np.pad(mfcc, ((0, 0), (0, pad)), mode='constant')
            return torch.from_numpy(mfcc).unsqueeze(0).float()  # (1, n_mfcc, T)
        except Exception:
            pass
    return torch.zeros(1, n_mfcc, target_len)


# ------------------------------------------------------------------ #
#  Dataset                                                             #
# ------------------------------------------------------------------ #
class DeepfakeDataset(Dataset):
    """
    Video-level dataset.

    Each sample represents one VIDEO (a directory of frames + audio.wav).
    Returns:
        face_seq  : (T, 3, H, W)
        lip_seq   : (T, 3, H, W)
        eye_seq   : (T, 3, H, W)
        mfcc      : (1, n_mfcc, T_audio)
        label     : scalar float32  (0=real, 1=fake)

    self.images is kept for backward-compat with evaluate.py/train.py
    but each entry is (representative_frame_path, label).
    """

    def __init__(self, root_dir: str, is_train: bool = False):
        self.root_dir = Path(root_dir)
        self.is_train = is_train
        self.face_extractor = FaceRegionExtractor()

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ])
        self.aug_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ])

        # Load inventory to map relative paths to original raw paths
        split_name = Path(root_dir).name  # 'train', 'val', or 'test'
        csv_path = Path("outputs/manifests") / f"{split_name}.csv"
        
        self.video_to_orig = {}
        if csv_path.exists():
            import csv
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    orig_path = row["video_path"]
                    video_id = make_video_id(orig_path)
                    self.video_to_orig[video_id] = orig_path

        # Load precomputed media metadata
        self.media_metadata = {}
        meta_json = Path("outputs/manifests/media_metadata.json")
        if meta_json.exists():
            import json
            with open(meta_json, 'r', encoding='utf-8') as f:
                self.media_metadata = json.load(f)

        # self.videos: list of (video_dir, label)
        # self.images: list of (repr_frame_path, label)   ← kept for compat
        self.videos = []
        self.images = []
        self._load_dataset()

    def encode_metadata(self, video_stem):
        orig_path = self.video_to_orig.get(video_stem)
        meta = self.media_metadata.get(orig_path, {}) if orig_path else {}
        
        # 13 Numeric features:
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
            norm_gop, norm_nb_frames, norm_rotation
        ]
        
        # Categorical maps (One-hot encoding):
        video_codec = meta.get("video_codec", "unknown")
        v_codec_idx = {"h264": 0, "hevc": 1, "vp9": 2, "mpeg4": 3}.get(video_codec, 4)
        v_codec_onehot = [0.0] * 5
        v_codec_onehot[v_codec_idx] = 1.0
        
        audio_codec = meta.get("audio_codec", "unknown")
        a_codec_idx = {"aac": 0, "mp3": 1, "pcm_s16le": 2, "opus": 3}.get(audio_codec, 4)
        a_codec_onehot = [0.0] * 5
        a_codec_onehot[a_codec_idx] = 1.0
        
        pix_fmt = meta.get("pix_fmt", "unknown")
        pix_idx = {"yuv420p": 0, "yuvj420p": 1}.get(pix_fmt, 2)
        pix_onehot = [0.0] * 3
        pix_onehot[pix_idx] = 1.0
        
        container = meta.get("container_format", "unknown")
        c_idx = 0 if "mp4" in container or "mov" in container or "m4a" in container else (1 if "avi" in container else 2)
        c_onehot = [0.0] * 3
        c_onehot[c_idx] = 1.0
        
        # Booleans:
        vfr = 1.0 if meta.get("vfr", False) else 0.0
        has_creation_time = 1.0 if meta.get("has_creation_time", False) else 0.0
        missing_audio = 1.0 if meta.get("missing_audio", True) else 0.0
        missing_video = 1.0 if meta.get("missing_video", True) else 0.0
        has_tags = 1.0 if meta.get("has_tags", False) else 0.0
        
        bool_feats = [vfr, has_creation_time, missing_audio, missing_video, has_tags]
        
        feats = numeric_feats + v_codec_onehot + a_codec_onehot + pix_onehot + c_onehot + bool_feats
        return torch.tensor(feats, dtype=torch.float32)


    def _load_dataset(self):
        temp_videos = []
        temp_images = []
        
        for label_val, subdir in [(0.0, "real"), (1.0, "fake")]:
            class_dir = self.root_dir / subdir
            if not class_dir.exists():
                continue
            # Each immediate sub-directory is one video's frame folder.
            # Falls back to a flat list of images if no subdirectory structure exists.
            video_dirs = sorted([d for d in class_dir.iterdir() if d.is_dir()])
            if video_dirs:
                for vdir in video_dirs:
                    frames = sorted(list(vdir.glob("*.jpg")))
                    if frames:
                        temp_videos.append((vdir, label_val))
                        # Pick the middle frame as representative
                        temp_images.append((str(frames[len(frames) // 2]), label_val))
            else:
                # Flat layout — each jpg is a separate sample; treat as 1-frame video
                for frame in sorted(class_dir.glob("*.jpg")):
                    temp_videos.append((frame.parent, label_val))
                    temp_images.append((str(frame), label_val))
                    
        # Balance dataset if training: Fake = 3 * Real
        if self.is_train:
            import random
            real_items = [(v, i) for v, i in zip(temp_videos, temp_images) if v[1] == 0.0]
            fake_items = [(v, i) for v, i in zip(temp_videos, temp_images) if v[1] == 1.0]
            
            real_count = len(real_items)
            if real_count > 0:
                target_fake_count = 3 * real_count
                if len(fake_items) > target_fake_count:
                    random.seed(42)  # For reproducibility
                    fake_items = random.sample(fake_items, target_fake_count)
            
            combined = real_items + fake_items
            random.seed(42)
            random.shuffle(combined)
            
            self.videos = [item[0] for item in combined]
            self.images = [item[1] for item in combined]
        else:
            self.videos = temp_videos
            self.images = temp_images

        # If FAST_VERIFY environment variable is set to 1, subset to a very small size for speed
        if os.environ.get("FAST_VERIFY") == "1":
            import random
            random.seed(42)
            real_indices = [idx for idx, v in enumerate(self.videos) if v[1] == 0.0]
            fake_indices = [idx for idx, v in enumerate(self.videos) if v[1] == 1.0]
            
            # Select max 8 reals and 24 fakes
            sel_real = random.sample(real_indices, min(8, len(real_indices))) if real_indices else []
            sel_fake = random.sample(fake_indices, min(24, len(fake_indices))) if fake_indices else []
            
            selected_indices = sorted(sel_real + sel_fake)
            self.videos = [self.videos[idx] for idx in selected_indices]
            self.images = [self.images[idx] for idx in selected_indices]

    def __len__(self):
        return len(self.videos)

    def _load_frame(self, path: Path):
        try:
            return Image.open(path).convert("RGB")
        except Exception:
            return Image.new("RGB", (IMG_SIZE, IMG_SIZE))

    def _frame_to_tensor(self, pil_img):
        t = self.aug_transform if self.is_train else self.transform
        # aug_transform and transform both start from a PIL image already
        return t(pil_img)

    def __getitem__(self, idx):
        video_dir, label = self.videos[idx]

        # ---- 1. Sample frames ----------------------------------------
        all_frames = sorted(list(video_dir.glob("*.jpg")))
        if not all_frames:
            all_frames = [video_dir]          # video_dir IS the frame path

        # Uniform sampling — repeat if fewer frames than needed
        n = len(all_frames)
        indices = [int(i * n / FRAMES_PER_VIDEO) for i in range(FRAMES_PER_VIDEO)]
        sampled = [all_frames[min(i, n - 1)] for i in indices]

        face_tensors, lip_tensors, eye_tensors = [], [], []

        for fp in sampled:
            pil = self._load_frame(fp)
            face_pil, lip_pil, eye_pil = self.face_extractor.extract(pil)
            face_tensors.append(self._frame_to_tensor(face_pil))
            lip_tensors.append(self._frame_to_tensor(lip_pil))
            eye_tensors.append(self._frame_to_tensor(eye_pil))

        # Stack → (T, C, H, W)
        face_seq = torch.stack(face_tensors)
        lip_seq  = torch.stack(lip_tensors)
        eye_seq  = torch.stack(eye_tensors)

        # ---- 2. Audio (MFCC) -----------------------------------------
        audio_path = video_dir / "audio.wav"
        mfcc = load_mfcc(audio_path)

        # ---- 3. Label ------------------------------------------------
        lbl = torch.tensor(label, dtype=torch.float32)

        # ---- 4. Metadata ---------------------------------------------
        metadata_tensor = self.encode_metadata(video_dir.name)

        return face_seq, lip_seq, eye_seq, mfcc, metadata_tensor, lbl
