import csv
import os
import cv2
import hashlib
import subprocess
import multiprocessing
from pathlib import Path

RAW_DIR = Path("data/raw")
CROPS_DIR = Path("data/processed/crops")
FRAMES_PER_VIDEO = 8
IMG_SIZE = 224
REGIONS = ["face", "eyes", "lips", "jaw"]


def make_video_id(video_path: str) -> str:
    """Unique id = filename stem + hash of the raw-relative path. Plain stems
    collide across ~2,185 FakeAVCeleb identities that reuse the same filenames
    (e.g. 00109.mp4 under many different id folders)."""
    p = Path(video_path)
    try:
        rel = p.relative_to(RAW_DIR)
    except ValueError:
        rel = p
    unique_hash = hashlib.md5(str(rel).replace("\\", "/").encode("utf-8")).hexdigest()[:10]
    return f"{p.stem}_{unique_hash}"


class FaceRegionExtractor:
    """Detects a face and crops 4 regions: full face, eyes (upper ~40%), lips
    (lower ~40%), and jawline/boundary strip (lower jaw + a margin below the
    chin, where face-swap blending seams tend to concentrate).

    Primary detector is MediaPipe Face Landmarker (468 landmarks -> precise
    region boxes). MTCNN is a fallback coarse-bbox detector for frames where
    the landmarker fails. Falls back to the full frame if both fail.

    Lazily initialized per worker process so the class stays picklable for
    multiprocessing.Pool.
    """

    def __init__(self):
        self._landmarker = None
        self._mtcnn = None
        self._init_attempted = False

    def _init_backends(self):
        if self._init_attempted:
            return
        self._init_attempted = True

        try:
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision
            import urllib.request, tempfile

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
                min_face_detection_confidence=0.4,
            )
            self._landmarker = mp_vision.FaceLandmarker.create_from_options(opts)
        except Exception:
            self._landmarker = None

        try:
            from mtcnn import MTCNN
            self._mtcnn = MTCNN()
        except Exception:
            self._mtcnn = None

    def _detect_box(self, frame_rgb, w, h):
        if self._landmarker is not None:
            try:
                import mediapipe as mp
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                result = self._landmarker.detect(mp_image)
                if result.face_landmarks:
                    lm = result.face_landmarks[0]
                    xs = [p.x * w for p in lm]
                    ys = [p.y * h for p in lm]
                    return (
                        max(0, int(min(xs))), max(0, int(min(ys))),
                        min(w, int(max(xs))), min(h, int(max(ys))),
                    )
            except Exception:
                pass

        if self._mtcnn is not None:
            try:
                dets = self._mtcnn.detect_faces(frame_rgb)
                if dets:
                    x, y, bw, bh = dets[0]["box"]
                    return (max(0, x), max(0, y), min(w, x + bw), min(h, y + bh))
            except Exception:
                pass

        return None

    def extract(self, frame_bgr):
        """Returns {region_name: BGR crop resized to IMG_SIZE}."""
        self._init_backends()
        h, w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        box = self._detect_box(frame_rgb, w, h)
        if box is None:
            box = (0, 0, w, h)

        x1, y1, x2, y2 = box
        fh = max(1, y2 - y1)

        crops_raw = {
            "face": frame_bgr[y1:y2, x1:x2],
            "eyes": frame_bgr[y1: y1 + int(fh * 0.4), x1:x2],
            "lips": frame_bgr[y1 + int(fh * 0.6): y2, x1:x2],
            "jaw": frame_bgr[y1 + int(fh * 0.75): min(h, y2 + int(fh * 0.15)), x1:x2],
        }

        out = {}
        for name in REGIONS:
            crop = crops_raw[name]
            if crop is None or crop.size == 0:
                crop = frame_bgr
            out[name] = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))
        return out


_extractor = None


def get_extractor():
    global _extractor
    if _extractor is None:
        _extractor = FaceRegionExtractor()
    return _extractor


def expected_files(out_dir: Path):
    return [out_dir / f"frame{i}_{r}.jpg" for i in range(FRAMES_PER_VIDEO) for r in REGIONS]


def process_video(item):
    video_path, split_name, label = item
    video_path = Path(video_path)
    video_id = make_video_id(str(video_path))
    out_dir = CROPS_DIR / split_name / label / video_id
    out_dir.mkdir(parents=True, exist_ok=True)

    audio_path = out_dir / "audio.wav"
    if audio_path.exists() and all(f.exists() for f in expected_files(out_dir)):
        return True

    success = True

    if not audio_path.exists():
        cmd_audio = [
            "ffmpeg", "-y", "-i", str(video_path),
            "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
            "-loglevel", "error", str(audio_path),
        ]
        try:
            subprocess.run(cmd_audio, check=True)
        except subprocess.CalledProcessError:
            # Some sources (e.g. this FFPP export) have video-only clips with
            # no audio stream at all - ffmpeg can't extract what isn't there.
            # Write a short silence placeholder instead of leaving audio.wav
            # missing, so this video doesn't get treated as incomplete and
            # re-attempted forever, and downstream Whisper gets valid (silent)
            # audio reflecting the true absence of sound.
            cmd_silence = [
                "ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono",
                "-t", "1", "-acodec", "pcm_s16le", "-loglevel", "error", str(audio_path),
            ]
            try:
                subprocess.run(cmd_silence, check=True)
            except subprocess.CalledProcessError:
                success = False

    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return False

    indices = [int(i * total / FRAMES_PER_VIDEO) for i in range(FRAMES_PER_VIDEO)]
    extractor = get_extractor()

    for i, idx in enumerate(indices):
        frame_done = all((out_dir / f"frame{i}_{r}.jpg").exists() for r in REGIONS)
        if frame_done:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            success = False
            continue
        crops = extractor.extract(frame)
        for region, crop in crops.items():
            out_path = out_dir / f"frame{i}_{region}.jpg"
            if not out_path.exists():
                cv2.imwrite(str(out_path), crop)

    cap.release()
    return success


def main():
    print("Phase 5: Extracting 8-frame / 4-region crops + audio (from raw video)...")

    splits = {
        "train": Path("outputs/manifests/train.csv"),
        "val": Path("outputs/manifests/val.csv"),
        "test": Path("outputs/manifests/test.csv"),
        "held_out": Path("outputs/manifests/held_out_crossdataset.csv"),
    }

    tasks = []
    for split_name, csv_path in splits.items():
        if not csv_path.exists():
            print(f"Warning: {csv_path} not found. Skipping.")
            continue
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            for row in reader:
                video_path = row[0]
                label = "fake" if row[1].lower() == "fake" else "real"
                tasks.append((video_path, split_name, label))

    if not tasks:
        print("No videos to process. Did you run phase 4?")
        return

    print(f"Extracting features for {len(tasks)} videos...")

    # Capped well below cpu_count(): each worker loads its own MediaPipe+MTCNN
    # models, which previously exhausted the pagefile at 19 workers on this
    # 16GB machine (especially with other memory-heavy processes running
    # concurrently, e.g. a large file transfer).
    num_cores = min(6, max(1, multiprocessing.cpu_count() - 1))
    print(f"Using {num_cores} cores processing in parallel...")

    processed = 0
    with multiprocessing.Pool(num_cores) as pool:
        for i, success in enumerate(pool.imap_unordered(process_video, tasks)):
            if success:
                processed += 1
            if (i + 1) % 500 == 0:
                print(f"Processed {i + 1}/{len(tasks)} videos...")

    print(f"Phase 5 completed! Successfully extracted features for {processed}/{len(tasks)} videos.")


if __name__ == "__main__":
    main()
