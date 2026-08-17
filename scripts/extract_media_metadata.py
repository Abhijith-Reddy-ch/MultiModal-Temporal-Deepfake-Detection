import csv
import json
import os
import subprocess
import multiprocessing
from pathlib import Path

# Absolute paths or relative paths in project root
MANIFESTS_DIR = Path("outputs/manifests")
TRAIN_CSV = MANIFESTS_DIR / "train.csv"
VAL_CSV = MANIFESTS_DIR / "val.csv"
TEST_CSV = MANIFESTS_DIR / "test.csv"
OUTPUT_JSON = MANIFESTS_DIR / "media_metadata.json"

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

def process_single_video(video_path_str):
    video_path = Path(video_path_str)
    if not video_path.exists():
        # Try relative to workspace
        video_path = Path(os.getcwd()) / video_path_str
        if not video_path.exists():
            return video_path_str, None
            
    metadata = {
        "file_size": os.path.getsize(video_path),
        "duration": 0.0,
        "nb_streams": 0,
        "container_format": "unknown",
        "video_codec": "unknown",
        "width": 0,
        "height": 0,
        "fps": 0.0,
        "video_bitrate": 0.0,
        "pix_fmt": "unknown",
        "color_space": "unknown",
        "nb_frames": 0,
        "rotation": 0.0,
        "vfr": False,
        "audio_codec": "unknown",
        "audio_bitrate": 0.0,
        "audio_sample_rate": 0.0,
        "audio_channels": 0,
        "gop_length": 0,
        "has_creation_time": False,
        "missing_audio": True,
        "missing_video": True,
        "has_tags": False
    }
    
    try:
        # Run ffprobe for format and streams
        cmd = ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", str(video_path)]
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(res.stdout)
        
        # Format info
        fmt_info = data.get("format", {})
        metadata["duration"] = float(fmt_info.get("duration", 0.0) or 0.0)
        metadata["nb_streams"] = int(fmt_info.get("nb_streams", 0) or 0)
        metadata["container_format"] = fmt_info.get("format_name", "unknown")
        
        fmt_tags = fmt_info.get("tags", {})
        if fmt_tags:
            metadata["has_tags"] = True
            # Check for creation time or modification time in tags
            if any(k in fmt_tags for k in ["creation_time", "modification_time", "com.apple.quicktime.creationdate"]):
                metadata["has_creation_time"] = True
                
        # Parse streams
        streams = data.get("streams", [])
        for s in streams:
            t = s.get("codec_type")
            if t == "video":
                metadata["missing_video"] = False
                metadata["video_codec"] = s.get("codec_name", "unknown")
                metadata["width"] = int(s.get("width", 0) or 0)
                metadata["height"] = int(s.get("height", 0) or 0)
                metadata["pix_fmt"] = s.get("pix_fmt", "unknown")
                metadata["nb_frames"] = int(s.get("nb_frames", 0) or 0)
                metadata["video_bitrate"] = float(s.get("bit_rate", 0.0) or 0.0)
                
                # FPS calculation
                avg_fr_val = parse_rate(s.get("avg_frame_rate", "0/0"))
                r_fr_val = parse_rate(s.get("r_frame_rate", "0/0"))
                metadata["fps"] = avg_fr_val if avg_fr_val > 0 else r_fr_val
                metadata["vfr"] = abs(avg_fr_val - r_fr_val) > 1e-4
                
                # Color space info
                color_prim = s.get("color_primaries", "")
                color_tr = s.get("color_transfer", "")
                color_sp = s.get("color_space", "")
                if color_prim or color_tr or color_sp:
                    metadata["color_space"] = f"{color_prim}/{color_tr}/{color_sp}"
                    
                # Video rotation
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
                
        # Fast GOP estimation using packet flags for the first 100 packets
        cmd_gop = ["ffprobe", "-v", "quiet", "-select_streams", "v:0", "-show_entries", "packet=flags", "-of", "csv=p=0", "-read_intervals", "%+#100", str(video_path)]
        res_gop = subprocess.run(cmd_gop, capture_output=True, text=True, check=True)
        flags = res_gop.stdout.strip().split()
        if flags:
            i_indices = [i for i, f in enumerate(flags) if 'K' in f]
            gop = i_indices[1] - i_indices[0] if len(i_indices) > 1 else len(flags)
            metadata["gop_length"] = gop
        else:
            metadata["gop_length"] = 0
            
    except Exception as e:
        # Keep defaults if ffprobe fails
        pass
        
    return video_path_str, metadata

def main():
    print("Media Metadata Extractor")
    
    # 1. Gather all video paths from CSV files
    video_paths = set()
    for csv_file in [TRAIN_CSV, VAL_CSV, TEST_CSV]:
        if not csv_file.exists():
            print(f"Warning: {csv_file} not found. Skipping.")
            continue
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for row in reader:
                if row:
                    video_paths.add(row[0])
                    
    print(f"Found {len(video_paths)} unique video paths in manifests.")
    
    # Load existing JSON if present to resume or append
    existing_data = {}
    if OUTPUT_JSON.exists():
        try:
            with open(OUTPUT_JSON, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
            print(f"Loaded {len(existing_data)} existing metadata records.")
        except Exception:
            pass
            
    # Filter out already processed paths
    to_process = [p for p in video_paths if p not in existing_data]
    print(f"Remaining paths to process: {len(to_process)}")
    
    if not to_process:
        print("All videos are already processed!")
        return
        
    # Process in parallel using a multiprocessing pool
    num_cores = max(1, multiprocessing.cpu_count() - 1)
    print(f"Running extraction on {num_cores} cores...")
    
    results = {}
    processed = 0
    
    try:
        with multiprocessing.Pool(num_cores) as pool:
            for path_str, meta in pool.imap_unordered(process_single_video, to_process):
                if meta:
                    results[path_str] = meta
                processed += 1
                if processed % 500 == 0 or processed == len(to_process):
                    print(f"Progress: {processed}/{len(to_process)} completed...")
                    # Save incremental progress
                    combined = {**existing_data, **results}
                    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
                        json.dump(combined, f, indent=2)
                        
    except KeyboardInterrupt:
        print("Interrupt received. Saving current progress...")
        
    # Save final results
    combined = {**existing_data, **results}
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(combined, f, indent=2)
    print(f"Extraction complete! Metadata saved to {OUTPUT_JSON}")

if __name__ == "__main__":
    main()
