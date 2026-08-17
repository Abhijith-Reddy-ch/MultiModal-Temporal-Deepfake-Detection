import base64
import os
import shutil
import tempfile
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.concurrency import run_in_threadpool
import uvicorn

from training.config import API_PORT, API_HOST
from training.infer import load_model, infer_video_file
from training.explainability import explain_video

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="DeepFake Detection API")

# Allow CORS for Next.js development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Triggering auto-reload for new deepfake_model.pth weights!
# Load model globally on startup
print("Loading model for API...")
try:
    model = load_model()
    print("Model loaded successfully.")
except Exception as e:
    print(f"Failed to load model: {e}")
    model = None

def get_file_type(file_path: str):
    ext = Path(file_path).suffix.lower()
    if ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
        return "video/mp4"
    elif ext in ['.wav', '.mp3', '.m4a', '.flac', '.ogg']:
        return "audio/wav"
    elif ext in ['.jpg', '.jpeg', '.png', '.webp']:
        return "image/jpeg"
    return "application/octet-stream"

@app.get("/health")
async def health_check():
    if model is not None:
        return {"status": "ok", "model_loaded": True}
    return {"status": "error", "model_loaded": False}

@app.post("/predict")
async def predict_file(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded or initialized correctly")
        
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name
        
    try:
        mime_type = get_file_type(tmp_path)
        
        if mime_type.startswith("video") or mime_type.startswith("application/octet-stream"):
            # Some videos might have generic mime types on Windows
            avg_prob, pred_label, conf, modality_scores, frame_probs, contributions, raw_meta, manipulation_breakdown = await run_in_threadpool(infer_video_file, model, Path(tmp_path))
            print(f"[PREDICT] Ingested file: {file.filename}")
            print(f"          Verdict: {pred_label} (prob: {avg_prob:.4f}, conf: {conf:.4f})")
            print(f"          Anomalies -> VFR: {raw_meta.get('vfr')}, Missing Audio: {raw_meta.get('missing_audio')}, Missing Created Time: {not raw_meta.get('has_creation_time')}")
            return {
                "type": "video",
                "prediction": pred_label,
                "confidence": conf,
                "fake_probability": avg_prob,
                "modality_scores": modality_scores,
                "frame_probabilities": frame_probs,
                "modality_contributions": contributions,
                "metadata": raw_meta,
                "manipulation_type_breakdown": manipulation_breakdown,
            }
        else:
            raise HTTPException(status_code=400, detail="Only video files are supported for prediction")
            
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

@app.post("/gradcam")
async def generate_gradcam(file: UploadFile = File(...)):
    """Grad-CAM (last DINOv2 block's CLS-to-patch attention, gradient-weighted
    from the real fake-logit) + fusion-transformer attention rollout +
    GMU gate, per plan.pdf Phase 7."""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded or initialized correctly")

    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    out_dir = Path(tempfile.mkdtemp(prefix="gradcam_"))
    try:
        result = await run_in_threadpool(explain_video, model, Path(tmp_path), out_dir)

        gradcam_images_b64 = {}
        for region, frames in result["gradcam_image_paths"].items():
            gradcam_images_b64[region] = {}
            for frame_idx, img_path in frames.items():
                with open(img_path, "rb") as f:
                    gradcam_images_b64[region][frame_idx] = base64.b64encode(f.read()).decode("utf-8")

        return {
            "fake_probability": result["fake_probability"],
            "gmu_gate_visual_weight": result["gmu_gate_visual_weight"],
            "gmu_gate_audio_weight": result["gmu_gate_audio_weight"],
            "manipulation_type_breakdown": result["manipulation_type_breakdown"],
            "attention_rollout_per_frame": result["attention_rollout_per_frame"],
            "gradcam_images": gradcam_images_b64,
        }
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        shutil.rmtree(out_dir, ignore_errors=True)

if __name__ == "__main__":
    uvicorn.run("backend.app:app", host=API_HOST, port=API_PORT, reload=True)
