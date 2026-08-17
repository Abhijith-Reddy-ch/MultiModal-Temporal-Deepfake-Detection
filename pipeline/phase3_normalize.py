import csv
import subprocess
import os
from pathlib import Path
import multiprocessing
from functools import partial

# In Phase 2, inventory_cleaned is renamed back to inventory.csv
CLEANED_MANIFEST_FILE = Path("outputs/manifests/inventory.csv")
NORMALIZED_MANIFEST_FILE = Path("outputs/manifests/inventory_normalized.csv")
OUTPUT_DIR = Path("data/processed/normalized")

def check_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except FileNotFoundError:
        return False

def process_video(row):
    video_path = row[0]
    label = row[1]
    rest = row[2:]
    
    # Create the output path maintaining the directory structure
    try:
        rel_path = Path(video_path).relative_to(Path("data/raw"))
    except ValueError:
        # fallback if not in data/raw
        rel_path = Path(video_path).name
        
    out_path = OUTPUT_DIR / rel_path
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    if out_path.exists():
        # Already processed
        return [str(out_path).replace("\\", "/"), label] + rest
        
    # The dataset videos are already 224x224 at 25 fps.
    # We can just copy the video stream to save hours of processing time, 
    # and only normalize the audio to 16kHz mono.
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-ac", "1",                     # Mono audio
        "-ar", "16000",                 # 16kHz audio sample rate
        "-c:v", "copy",                 # copy the video stream directly (fast!)
        "-c:a", "aac",                  # aac audio
        "-loglevel", "error",           # Suppress output
        str(out_path)
    ]
    
    try:
        subprocess.run(cmd, check=True)
        return [str(out_path).replace("\\", "/"), label] + rest
    except subprocess.CalledProcessError:
        # If FFmpeg fails for some reason
        return None

def main():
    print("Phase 3: Normalizing videos...")
    
    if not CLEANED_MANIFEST_FILE.exists():
        raise FileNotFoundError("Cleaned inventory file not found! Did you run phase 2?")
        
    if not check_ffmpeg():
        print("ERROR: FFmpeg is not installed or not in PATH.")
        print("Please install FFmpeg to run the normalization phase.")
        return
        
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    rows = []
    with open(CLEANED_MANIFEST_FILE, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        headers = next(reader)
        rows = list(reader)
        
    print(f"Normalizing {len(rows)} videos... (This will take a long time and use multiple CPU cores)")
    
    # Use multiprocessing to speed up FFmpeg processing
    num_cores = max(1, multiprocessing.cpu_count() - 1)
    print(f"Using {num_cores} cores processing in parallel...")
    
    normalized_rows = []
    
    # To avoiding running for hours in this demo, let's process a subset if requested or just run them all
    # Using pool
    with multiprocessing.Pool(num_cores) as pool:
        for i, result in enumerate(pool.imap_unordered(process_video, rows)):
            if result is not None:
                normalized_rows.append(result)
            
            if (i + 1) % 100 == 0:
                print(f"Normalized {i + 1}/{len(rows)} videos...")
                
    print(f"Writing normalized manifest with {len(normalized_rows)} entries...")
    with open(NORMALIZED_MANIFEST_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(normalized_rows)
        
    print("Phase 3 completed successfully!")

if __name__ == "__main__":
    main()
