"""One-off diagnostic: run Grad-CAM (training/explainability.py) on the
handful of DFDC held-out fakes the canonical Stage B model is MOST
confidently wrong about (from outputs/predictions/dfdc_error_analysis.csv),
to see what visual regions/frames it's actually keying on when it decides
"real" with near-zero fake probability on an actual fake."""
import os
import sys
import csv
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'training'))

from training.infer import load_model
from training.explainability import explain_video

N_VIDEOS = 6
CSV_PATH = "outputs/predictions/dfdc_error_analysis.csv"
OUT_ROOT = "outputs/misclassified/dfdc_gradcam_spotcheck"


def main():
    rows = list(csv.DictReader(open(CSV_PATH, encoding="utf-8")))
    wrong_fakes = [r for r in rows if r["label"] == "fake" and r["correct"] == "False"]
    wrong_fakes.sort(key=lambda r: float(r["prob"]))
    targets = wrong_fakes[:N_VIDEOS]

    print(f"Running Grad-CAM on {len(targets)} most-confidently-wrong DFDC fakes:")
    for r in targets:
        print(f"  {r['video_path']}  prob={float(r['prob']):.4f}  original={r['original']}")

    model = load_model()

    os.makedirs(OUT_ROOT, exist_ok=True)
    for r in targets:
        video_path = r["video_path"]
        vid_id = os.path.splitext(os.path.basename(video_path))[0]
        out_dir = os.path.join(OUT_ROOT, vid_id)
        print(f"\n--- {video_path} (orig fake_prob {float(r['prob']):.4f}) ---")
        try:
            result = explain_video(model, video_path, out_dir)
            print(f"  Re-scored fake_probability: {result['fake_probability']:.4f}")
            print(f"  GMU gate (visual weight): {result['gmu_gate_visual_weight']:.4f}")
            print(f"  Manipulation-type breakdown (top 3): "
                  f"{sorted(result['manipulation_type_breakdown'].items(), key=lambda kv: -kv[1])[:3]}")
            print(f"  Grad-CAM images written to: {out_dir}")
        except Exception as e:
            print(f"  FAILED: {e}")

    print(f"\nDone. All heatmaps under {OUT_ROOT}/<video_id>/gradcam_frame*_<region>.jpg")


if __name__ == "__main__":
    main()
