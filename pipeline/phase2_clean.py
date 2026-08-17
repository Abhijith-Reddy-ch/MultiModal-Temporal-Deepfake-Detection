import csv
import cv2
import os
import multiprocessing
from pathlib import Path

MANIFEST_FILE = Path("outputs/manifests/inventory.csv")
CLEANED_MANIFEST_FILE = Path("outputs/manifests/inventory_cleaned.csv")

def is_video_valid(path):
    if not os.path.exists(path):
        return False

    try:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return False

        # Check first 5 frames to ensure video is readable
        for _ in range(5):
            ret, frame = cap.read()
            if not ret:
                cap.release()
                return False

        cap.release()
        return True
    except Exception:
        return False

def check_row(row):
    return row, is_video_valid(row[0])

def main():
    print("Phase 2: Cleaning dataset...")
    if not MANIFEST_FILE.exists():
        raise FileNotFoundError("Inventory file not found! Did you run phase 1?")

    with open(MANIFEST_FILE, 'r', encoding='utf-8') as f_in:
        reader = csv.reader(f_in)
        headers = next(reader)
        rows = list(reader)

    valid_count = 0
    invalid_count = 0
    processed = 0

    num_cores = max(1, multiprocessing.cpu_count() - 1)
    print(f"Validating {len(rows)} videos using {num_cores} cores in parallel...")

    with open(CLEANED_MANIFEST_FILE, 'w', newline='', encoding='utf-8') as f_out:
        writer = csv.writer(f_out)
        writer.writerow(headers)

        with multiprocessing.Pool(num_cores) as pool:
            for row, valid in pool.imap(check_row, rows, chunksize=32):
                if valid:
                    writer.writerow(row)
                    valid_count += 1
                else:
                    invalid_count += 1

                processed += 1
                if processed % 1000 == 0:
                    print(f"Processed {processed}/{len(rows)} videos... ({invalid_count} corrupted found)")

    print(f"Cleaning complete! Found {valid_count} valid videos and {invalid_count} corrupted/missing videos.")
    print("Replacing old manifest with cleaned manifest...")

    # Replace the original inventory with the cleaned one
    os.replace(CLEANED_MANIFEST_FILE, MANIFEST_FILE)
    print("Phase 2 completed successfully.")

if __name__ == "__main__":
    main()
