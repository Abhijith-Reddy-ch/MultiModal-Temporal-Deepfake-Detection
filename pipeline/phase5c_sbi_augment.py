"""
Phase 5c: Self-Blended Images (SBI) augmentation (Shiohara & Yamasaki, CVPR 2022).

Generates pseudo-fake training samples from REAL train-split videos only, by
blending two differently color/quality-transformed copies of the same face
across a soft, landmark-derived mask with a smooth random warp on one copy.
This teaches the model a generator-agnostic "blending boundary" artifact
instead of any specific dataset's generator fingerprint - the aim is better
generalization to deepfake methods never seen in training (see DFDC held-out
cross-dataset gap).

Reads data/processed/crops/train/real/<video_id>/ (phase5_extract.py output).
Writes data/processed/crops/train/fake_sbi/<sbi_id>/ + outputs/manifests/train_sbi.csv.

Never touches val/test/held_out - SBI is a training-only regularizer.
"""
import argparse
import csv
import hashlib
import multiprocessing
import os
import random
import shutil
import tempfile
import urllib.request
from pathlib import Path

import cv2
import numpy as np

RAW_DIR = Path("data/raw")
CROPS_DIR = Path("data/processed/crops")
TRAIN_MANIFEST = Path("outputs/manifests/train.csv")
SBI_MANIFEST = Path("outputs/manifests/train_sbi.csv")

FRAMES_PER_VIDEO = 8
IMG_SIZE = 224
REGIONS = ["face", "eyes", "lips", "jaw"]


def make_video_id(video_path: str) -> str:
    """Must match pipeline/phase5_extract.py's scheme exactly (see project's
    hash-collision fix - duplicated intentionally per this project's
    established convention rather than shared-imported)."""
    p = Path(video_path)
    try:
        rel = p.relative_to(RAW_DIR)
    except ValueError:
        rel = p
    unique_hash = hashlib.md5(str(rel).replace("\\", "/").encode("utf-8")).hexdigest()[:10]
    return f"{p.stem}_{unique_hash}"


def make_sbi_id(video_path: str) -> str:
    """Distinct id from the real entry's video_id so crop dirs never collide."""
    return make_video_id(f"{video_path}::sbi")


def derive_regions(face_bgr):
    """Same proportional slicing as phase5_extract.py::FaceRegionExtractor.extract,
    applied to an already-cropped+blended face image (fh = face image height)."""
    h, w = face_bgr.shape[:2]
    crops_raw = {
        "face": face_bgr,
        "eyes": face_bgr[0: int(h * 0.4), 0:w],
        "lips": face_bgr[int(h * 0.6): h, 0:w],
        "jaw": face_bgr[int(h * 0.75): h, 0:w],
    }
    out = {}
    for name in REGIONS:
        crop = crops_raw[name]
        if crop is None or crop.size == 0:
            crop = face_bgr
        out[name] = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))
    return out


class LandmarkDetector:
    """Lazily-initialized MediaPipe FaceLandmarker, picklable for
    multiprocessing.Pool (mirrors phase5_extract.py::FaceRegionExtractor)."""

    def __init__(self):
        self._landmarker = None
        self._init_attempted = False

    def _init(self):
        if self._init_attempted:
            return
        self._init_attempted = True
        try:
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision

            model_url = (
                "https://storage.googleapis.com/mediapipe-models/"
                "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
            )
            cache_path = os.path.join(tempfile.gettempdir(), "face_landmarker.task")
            if not os.path.exists(cache_path):
                urllib.request.urlretrieve(model_url, cache_path)

            base_opts = mp_python.BaseOptions(model_asset_path=cache_path)
            opts = mp_vision.FaceLandmarkerOptions(
                base_options=base_opts,
                num_faces=1,
                min_face_detection_confidence=0.3,
            )
            self._landmarker = mp_vision.FaceLandmarker.create_from_options(opts)
        except Exception:
            self._landmarker = None

    def landmarks_xy(self, frame_bgr):
        """Returns Nx2 float32 pixel-space landmark array, or None if detection fails."""
        self._init()
        if self._landmarker is None:
            return None
        try:
            import mediapipe as mp
            h, w = frame_bgr.shape[:2]
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            result = self._landmarker.detect(mp_image)
            if not result.face_landmarks:
                return None
            lm = result.face_landmarks[0]
            pts = np.array([[p.x * w, p.y * h] for p in lm], dtype=np.float32)
            return pts
        except Exception:
            return None


_detector = None


def get_detector():
    global _detector
    if _detector is None:
        _detector = LandmarkDetector()
    return _detector


def sample_recipe(seed: int) -> dict:
    """One SBI recipe per video, reused across all its frames - mirrors the
    per-video-consistent augmentation pattern in training/raw_crops_dataset.py
    (a blend style should reflect the pseudo-forgery's "pipeline", not vary
    frame-to-frame)."""
    rng = random.Random(seed)
    return {
        "brightness_a": rng.uniform(0.85, 1.15),
        "contrast_a": rng.uniform(0.85, 1.15),
        "saturation_a": rng.uniform(0.85, 1.15),
        "brightness_b": rng.uniform(0.85, 1.15),
        "contrast_b": rng.uniform(0.85, 1.15),
        "saturation_b": rng.uniform(0.85, 1.15),
        "b_blur_sigma": rng.uniform(0.0, 1.5),
        "mask_dilate_px": rng.randint(2, 10),
        "mask_blur_sigma": rng.uniform(3.0, 9.0),
        "warp_strength": rng.uniform(1.5, 6.0),
        "warp_field_seed": rng.randint(0, 2 ** 31 - 1),
    }


def adjust_color(img_bgr, brightness, contrast, saturation):
    img = img_bgr.astype(np.float32)
    img = (img - 127.5) * contrast + 127.5
    img = img * brightness
    hsv = cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] *= saturation
    hsv = np.clip(hsv, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def random_displacement_field(h, w, strength, seed):
    """Smooth random warp: low-res random offsets, upsampled + blurred, so the
    deformation is gentle/continuous rather than pixel-noise (mimics the slight
    geometric misalignment inherent to blending two different frames/sources)."""
    rng = np.random.RandomState(seed)
    low_res = 8
    field_x = rng.uniform(-strength, strength, (low_res, low_res)).astype(np.float32)
    field_y = rng.uniform(-strength, strength, (low_res, low_res)).astype(np.float32)
    field_x = cv2.resize(field_x, (w, h), interpolation=cv2.INTER_CUBIC)
    field_y = cv2.resize(field_y, (w, h), interpolation=cv2.INTER_CUBIC)
    field_x = cv2.GaussianBlur(field_x, (0, 0), sigmaX=w / 16.0)
    field_y = cv2.GaussianBlur(field_y, (0, 0), sigmaX=w / 16.0)

    grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    map_x = grid_x + field_x
    map_y = grid_y + field_y
    return map_x, map_y


def build_blend_mask(landmarks_xy, h, w, dilate_px, blur_sigma):
    mask = np.zeros((h, w), dtype=np.uint8)
    pts = landmarks_xy.astype(np.int32)
    hull = cv2.convexHull(pts)
    cv2.fillConvexPoly(mask, hull, 255)
    if dilate_px > 0:
        kernel = np.ones((dilate_px, dilate_px), np.uint8)
        mask = cv2.dilate(mask, kernel)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=blur_sigma)
    return (mask.astype(np.float32) / 255.0)[..., None]  # (H, W, 1)


def blend_frame(face_bgr, recipe: dict, detector: LandmarkDetector):
    """Returns the blended face image, or None if landmarks couldn't be found
    (caller falls back to skipping this frame's SBI variant)."""
    landmarks = detector.landmarks_xy(face_bgr)
    if landmarks is None:
        return None

    h, w = face_bgr.shape[:2]

    copy_a = adjust_color(face_bgr, recipe["brightness_a"], recipe["contrast_a"], recipe["saturation_a"])
    copy_b = adjust_color(face_bgr, recipe["brightness_b"], recipe["contrast_b"], recipe["saturation_b"])
    if recipe["b_blur_sigma"] > 0.05:
        copy_b = cv2.GaussianBlur(copy_b, (0, 0), sigmaX=recipe["b_blur_sigma"])

    map_x, map_y = random_displacement_field(h, w, recipe["warp_strength"], recipe["warp_field_seed"])
    copy_b_warped = cv2.remap(copy_b, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    mask = build_blend_mask(landmarks, h, w, recipe["mask_dilate_px"], recipe["mask_blur_sigma"])

    blended = mask * copy_b_warped.astype(np.float32) + (1 - mask) * copy_a.astype(np.float32)
    return np.clip(blended, 0, 255).astype(np.uint8)


def process_video(item):
    video_path, real_video_id = item
    real_dir = CROPS_DIR / "train" / "real" / real_video_id
    sbi_id = make_sbi_id(video_path)
    out_dir = CROPS_DIR / "train" / "fake_sbi" / sbi_id

    expected = [out_dir / f"frame{i}_{r}.jpg" for i in range(FRAMES_PER_VIDEO) for r in REGIONS]
    if (out_dir / "audio.wav").exists() and all(f.exists() for f in expected):
        return True, video_path

    out_dir.mkdir(parents=True, exist_ok=True)
    recipe = sample_recipe(seed=int(hashlib.md5(sbi_id.encode("utf-8")).hexdigest()[:8], 16))
    detector = get_detector()

    ok_any = False
    for i in range(FRAMES_PER_VIDEO):
        face_path = real_dir / f"frame{i}_face.jpg"
        face_img = cv2.imread(str(face_path))
        if face_img is None:
            continue
        blended = blend_frame(face_img, recipe, detector)
        if blended is None:
            continue
        regions = derive_regions(blended)
        for region, crop in regions.items():
            cv2.imwrite(str(out_dir / f"frame{i}_{region}.jpg"), crop)
        ok_any = True

    if ok_any:
        real_audio = real_dir / "audio.wav"
        if real_audio.exists():
            shutil.copyfile(real_audio, out_dir / "audio.wav")

    return ok_any, video_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="Cap number of real videos processed (0 = no cap, for smoke-testing)")
    args = parser.parse_args()

    print("Phase 5c: Generating Self-Blended Images (SBI) pseudo-fakes from real train videos...")

    if not TRAIN_MANIFEST.exists():
        print(f"Warning: {TRAIN_MANIFEST} not found. Did you run phase 4?")
        return

    real_rows = {}  # video_path -> (original_type, race, gender), for carrying the
                    # source real video's metadata into the SBI manifest row below
    with open(TRAIN_MANIFEST, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["label"].lower() == "real":
                real_rows[row["video_path"]] = (
                    row.get("original_type", ""),
                    row.get("race", ""),
                    row.get("gender", ""),
                )

    # Checkpoint/resume: skip videos already logged in a prior (possibly
    # interrupted) run, so stopping this script partway through and rerunning
    # it later never redoes finished work or loses progress.
    already_logged = set()
    manifest_exists = SBI_MANIFEST.exists()
    if manifest_exists:
        with open(SBI_MANIFEST, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                already_logged.add(row["video_path"].rsplit("::sbi", 1)[0])

    video_paths = [vp for vp in real_rows.keys() if vp not in already_logged]
    if args.limit > 0:
        video_paths = video_paths[: args.limit]

    tasks = [(vp, make_video_id(vp)) for vp in video_paths]
    tasks = [(vp, vid) for vp, vid in tasks if (CROPS_DIR / "train" / "real" / vid).exists()]

    if already_logged:
        print(f"Resuming: {len(already_logged)} videos already logged in {SBI_MANIFEST}, skipping.")

    if not tasks:
        print("Nothing left to do (all real train videos with crops are already logged).")
        return

    print(f"Generating SBI pseudo-fakes for {len(tasks)} real videos...")

    num_cores = min(6, max(1, multiprocessing.cpu_count() - 1))
    print(f"Using {num_cores} cores processing in parallel...")

    SBI_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if manifest_exists else "w"
    with open(SBI_MANIFEST, mode, newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not manifest_exists:
            writer.writerow(["video_path", "label", "original_type", "race", "gender", "source", "manipulation_type"])

        succeeded = 0
        with multiprocessing.Pool(num_cores) as pool:
            for i, (success, video_path) in enumerate(pool.imap_unordered(process_video, tasks)):
                if success:
                    original_type, race, gender = real_rows[video_path]
                    writer.writerow([f"{video_path}::sbi", "fake", original_type, race, gender, "SBI", "unknown_fake"])
                    f.flush()  # checkpoint: every completed video is durable on disk
                                # immediately, so killing this process anytime only
                                # loses at most the one video in flight
                    succeeded += 1
                if (i + 1) % 200 == 0:
                    print(f"Processed {i + 1}/{len(tasks)} videos...")

    print(f"Phase 5c completed! Generated {succeeded}/{len(tasks)} SBI pseudo-fakes -> {SBI_MANIFEST}")

    print(f"Phase 5c completed! Generated {len(succeeded_paths)}/{len(tasks)} SBI pseudo-fakes -> {SBI_MANIFEST}")


if __name__ == "__main__":
    main()
