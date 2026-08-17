"""
Inference for the plan.pdf architecture (DINOv2 + Whisper + cross-region
attention + GMU + classifier). Loads whichever final model is selected -
Stage A (frozen backbone) by default, or Stage B (LoRA fine-tuned) if its
artifacts are present and MODEL_STAGE="B" is set.

Output contract kept compatible with backend/app.py's existing /predict
response shape (modality_scores, modality_contributions, frame_probabilities,
metadata) - only the modality keys changed: this architecture's GMU gate is a
genuine visual-vs-audio signal (not the old model's face/lip/eye/audio region
heads, which don't exist in this architecture).
"""
import os
import sys
import json
import shutil
import tempfile
import subprocess
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

sys.path.append(os.getcwd())

import whisper
from training.config import MODELS_DIR
from training.model import DeepfakeClassifier, MANIPULATION_TYPES, load_classifier_state_dict
from pipeline.phase5_extract import FaceRegionExtractor, REGIONS, FRAMES_PER_VIDEO

DEFAULT_THRESHOLD = 0.990  # picked on VALIDATION by maximizing balanced accuracy,
# not F1 - see training/evaluate_stage_b.py's pick_threshold_on_val(). Recomputed
# 2026-08-17 for the canonical (attempt #2 recipe, corrected identity-safe split)
# checkpoint currently deployed at models/stageB_best_*; MUST be re-derived again
# any time the deployed checkpoint changes (this exact bug - a stale threshold
# left over from a prior checkpoint - was caught live during a demo test, see
# DFDC_GENERALIZATION_INVESTIGATION.md). F1-maximization on this project's
# ~20-85:1 fake:real imbalance previously produced a threshold of ~0.10, which
# misclassified 25-45% of real videos in practice (near-1.0 AUC still held,
# since AUC is rank-based and doesn't depend on threshold at all - the model
# was fine, the operating point wasn't). Re-derive via
# training/evaluate_stage_b.py if the model is retrained.
DINO_DIM = 384
IMG_SIZE = 224

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]

MODEL_STAGE = os.environ.get("MODEL_STAGE", "B")  # "A" (frozen) or "B" (LoRA fine-tuned)
# Stage B (augmentation + weight_decay=1e-4) selected as final: best DFDC
# cross-dataset AUC (0.7105) among 4 candidates tried - see project memory
# for the full comparison. Still a large gap vs 0.996 in-distribution AUC;
# not a full fix, but the best available.

_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
])


# ------------------------------------------------------------------ #
#  ffprobe metadata extraction (unchanged from the legacy inference    #
#  pipeline - still a 34-dim ffprobe-derived vector, same encoding     #
#  used everywhere else in this project).                              #
# ------------------------------------------------------------------ #
def parse_rate(rate_str):
    if not rate_str:
        return 0.0
    if "/" in rate_str:
        try:
            num, den = map(float, rate_str.split("/"))
            return num / den if den > 0 else 0.0
        except ValueError:
            return 0.0
    try:
        return float(rate_str)
    except ValueError:
        return 0.0


def extract_raw_metadata(video_path: Path):
    metadata = {
        "file_size": 0.0, "duration": 0.0, "nb_streams": 0, "container_format": "unknown",
        "video_codec": "unknown", "width": 0, "height": 0, "fps": 0.0, "video_bitrate": 0.0,
        "pix_fmt": "unknown", "color_space": "unknown", "nb_frames": 0, "rotation": 0.0,
        "vfr": False, "audio_codec": "unknown", "audio_bitrate": 0.0, "audio_sample_rate": 0.0,
        "audio_channels": 0, "gop_length": 0, "has_creation_time": False,
        "missing_audio": True, "missing_video": True, "has_tags": False,
    }
    if not video_path.exists():
        return metadata
    metadata["file_size"] = os.path.getsize(video_path)
    try:
        cmd = ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", str(video_path)]
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(res.stdout)
        fmt_info = data.get("format", {})
        metadata["duration"] = float(fmt_info.get("duration", 0.0) or 0.0)
        metadata["nb_streams"] = int(fmt_info.get("nb_streams", 0) or 0)
        metadata["container_format"] = fmt_info.get("format_name", "unknown")
        fmt_tags = fmt_info.get("tags", {})
        if fmt_tags:
            metadata["has_tags"] = True
            if any(k in fmt_tags for k in ["creation_time", "modification_time", "com.apple.quicktime.creationdate"]):
                metadata["has_creation_time"] = True
        for s in data.get("streams", []):
            t = s.get("codec_type")
            if t == "video":
                metadata["missing_video"] = False
                metadata["video_codec"] = s.get("codec_name", "unknown")
                metadata["width"] = int(s.get("width", 0) or 0)
                metadata["height"] = int(s.get("height", 0) or 0)
                metadata["pix_fmt"] = s.get("pix_fmt", "unknown")
                metadata["nb_frames"] = int(s.get("nb_frames", 0) or 0)
                metadata["video_bitrate"] = float(s.get("bit_rate", 0.0) or 0.0)
                avg_fr = parse_rate(s.get("avg_frame_rate", "0/0"))
                r_fr = parse_rate(s.get("r_frame_rate", "0/0"))
                metadata["fps"] = avg_fr if avg_fr > 0 else r_fr
                metadata["vfr"] = abs(avg_fr - r_fr) > 1e-4
                v_tags = s.get("tags", {})
                if v_tags and "rotate" in v_tags:
                    metadata["rotation"] = float(v_tags["rotate"])
                for side_data in s.get("side_data_list", []):
                    if "rotation" in side_data:
                        metadata["rotation"] = float(side_data["rotation"])
            elif t == "audio":
                metadata["missing_audio"] = False
                metadata["audio_codec"] = s.get("codec_name", "unknown")
                metadata["audio_bitrate"] = float(s.get("bit_rate", 0.0) or 0.0)
                metadata["audio_sample_rate"] = float(s.get("sample_rate", 0.0) or 0.0)
                metadata["audio_channels"] = int(s.get("channels", 0) or 0)
        cmd_gop = ["ffprobe", "-v", "quiet", "-select_streams", "v:0", "-show_entries", "packet=flags",
                   "-of", "csv=p=0", "-read_intervals", "%+#100", str(video_path)]
        res_gop = subprocess.run(cmd_gop, capture_output=True, text=True, check=True)
        flags = res_gop.stdout.strip().split()
        if flags:
            i_indices = [i for i, f in enumerate(flags) if "K" in f]
            metadata["gop_length"] = i_indices[1] - i_indices[0] if len(i_indices) > 1 else len(flags)
    except Exception:
        pass
    return metadata


def encode_raw_metadata(meta) -> torch.Tensor:
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
    return torch.tensor(feats, dtype=torch.float32).unsqueeze(0)


# ------------------------------------------------------------------ #
#  Model loading                                                       #
# ------------------------------------------------------------------ #
class LoadedModel:
    """Bundles whatever's needed to run inference: DINOv2, Whisper encoder
    (both frozen for Stage A, LoRA-wrapped for Stage B), and the classifier."""

    def __init__(self, dinov2, whisper_encoder, classifier, device, stage, n_mels):
        self.dinov2 = dinov2
        self.whisper_encoder = whisper_encoder
        self.classifier = classifier
        self.device = device
        self.stage = stage
        self.n_mels = n_mels


def load_model() -> LoadedModel:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dinov2_base = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    whisper_model = whisper.load_model("tiny", device=device)
    n_mels = whisper_model.dims.n_mels

    stage_b_dinov2_dir = MODELS_DIR / "stageB_best_dinov2_lora"
    stage_b_whisper_dir = MODELS_DIR / "stageB_best_whisper_lora"
    stage_b_classifier_path = MODELS_DIR / "stageB_best_classifier.pth"
    stage_a_path = MODELS_DIR / "stageA_best_model.pth"

    use_stage_b = MODEL_STAGE.upper() == "B" and stage_b_dinov2_dir.exists() and stage_b_classifier_path.exists()

    if use_stage_b:
        from peft import PeftModel
        dinov2 = PeftModel.from_pretrained(dinov2_base, str(stage_b_dinov2_dir))
        whisper_encoder = PeftModel.from_pretrained(whisper_model.encoder, str(stage_b_whisper_dir))
        classifier_path = stage_b_classifier_path
        stage = "B"
    else:
        dinov2 = dinov2_base
        whisper_encoder = whisper_model.encoder
        classifier_path = stage_a_path
        stage = "A"

    dinov2.eval().to(device)
    whisper_encoder.eval().to(device)

    classifier = DeepfakeClassifier().to(device)
    if classifier_path.exists():
        load_classifier_state_dict(
            classifier, torch.load(classifier_path, map_location=device, weights_only=False), label=str(classifier_path)
        )
        print(f"Loaded Stage {stage} classifier from {classifier_path}")
    else:
        print(f"WARNING: no classifier checkpoint found at {classifier_path} - using random init")
    classifier.eval()

    print(f"Inference model ready (Stage {stage}, device={device})")
    return LoadedModel(dinov2, whisper_encoder, classifier, device, stage, n_mels)


# ------------------------------------------------------------------ #
#  Video inference                                                      #
# ------------------------------------------------------------------ #
def infer_video_file(model: LoadedModel, video_path: Path, threshold: float = DEFAULT_THRESHOLD):
    device = model.device
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        # 1. Extract audio (fixed 16kHz mono; silent placeholder if the
        #    source has no audio stream at all - same as pipeline/phase5_extract.py)
        audio_path = tmp_dir / "audio.wav"
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(video_path), "-vn", "-acodec", "pcm_s16le",
             "-ar", "16000", "-ac", "1", "-loglevel", "error", str(audio_path)],
            check=False,
        )
        if not audio_path.exists():
            subprocess.run(
                ["ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono",
                 "-t", "1", "-acodec", "pcm_s16le", "-loglevel", "error", str(audio_path)],
                check=False,
            )

        # 2. Sample 8 frames, extract 4 regions each via the same extractor
        #    pipeline/phase5_extract.py uses (MediaPipe primary, MTCNN fallback)
        import cv2
        cap = cv2.VideoCapture(str(video_path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release()
            raise ValueError(f"Could not read frames from {video_path}")

        indices = [int(i * total / FRAMES_PER_VIDEO) for i in range(FRAMES_PER_VIDEO)]
        extractor = FaceRegionExtractor()
        region_tensors = []  # will hold 8*4=32 tensors in frame-major, region-minor order
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                frame = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
            crops = extractor.extract(frame)  # {"face":..,"eyes":..,"lips":..,"jaw":..} BGR arrays
            for region in REGIONS:
                pil = Image.fromarray(cv2.cvtColor(crops[region], cv2.COLOR_BGR2RGB))
                region_tensors.append(_transform(pil))
        cap.release()

        images = torch.stack(region_tensors).to(device)  # [32, 3, 224, 224]

        # 3. Audio -> mel spectrogram
        audio = whisper.load_audio(str(audio_path))
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio, n_mels=model.n_mels)
        mel = mel.unsqueeze(0).to(device)

        # 4. Metadata. raw_meta is still reported in the API response (it's
        # genuinely useful forensic info), but the classifier itself gets an
        # all-zero vector: media_metadata.json was broken for this project's
        # entire training run (kept the old, deleted phase3 paths), so the
        # metadata_encoder head was trained exclusively on zeros and never
        # learned to handle real ffprobe values. Feeding it real metadata at
        # serving time would push it into an input space it was never
        # trained on - pure train/serve skew, not a genuine forensic signal
        # the model actually knows how to use.
        raw_meta = extract_raw_metadata(video_path)
        metadata_tensor = torch.zeros((1, 34), dtype=torch.float32, device=device)

        # 5. Forward pass
        with torch.no_grad():
            cls_tokens = model.dinov2(images)  # [32, 384]
            visual = cls_tokens.view(1, FRAMES_PER_VIDEO, len(REGIONS), DINO_DIM)

            hidden = model.whisper_encoder(mel)
            hidden = hidden.transpose(1, 2)
            audio_feat = F.adaptive_avg_pool1d(hidden, FRAMES_PER_VIDEO).transpose(1, 2)  # [1, 8, 384]

            logits, feats = model.classifier(visual, audio_feat, metadata=metadata_tensor, return_features=True)
            prob = torch.sigmoid(logits).item()
            aux_probs = torch.softmax(feats["aux_forgery_logits"], dim=-1).squeeze(0).cpu().tolist()
            # Real GMU gate from the actual forward pass (post fusion-transformer) -
            # this architecture's genuine interpretable visual-vs-audio signal,
            # replacing the old model's face/lip/eye/audio region heads (which
            # don't exist here). See plan.pdf's explainability phase.
            gmu_gate = feats["gmu_gate"].item()

        modality_scores = {"visual": gmu_gate, "audio": 1.0 - gmu_gate}
        modality_contributions = {"visual": gmu_gate * 100, "audio": (1.0 - gmu_gate) * 100}

        manipulation_breakdown = dict(zip(MANIPULATION_TYPES, aux_probs))

        pred_label = "Fake" if prob >= threshold else "Real"
        conf = prob if prob >= threshold else 1 - prob

        return prob, pred_label, conf, modality_scores, [prob] * FRAMES_PER_VIDEO, modality_contributions, raw_meta, manipulation_breakdown

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    model = load_model()
    if len(sys.argv) > 1:
        result = infer_video_file(model, Path(sys.argv[1]))
        prob, label, conf, mods, frame_probs, contribs, raw_meta, manip = result
        print(f"Result: {label} ({conf*100:.1f}%)")
        print(f"Modality scores: {mods}")
        print(f"Contributions: {contribs}")
        print(f"Likely manipulation type: {max(manip, key=manip.get)}")
        print(f"Metadata: {raw_meta}")
