import requests
import os
from pathlib import Path

url = "http://127.0.0.1:8000/predict"
# Picked a sample video from the dataset
video_path = r"d:\deepfake_detection\deepfake_detection_project\data\raw\FakeAVCeleb_v1.2\FakeVideo-FakeAudio\Caucasian (European)\women\id03941\00021_id03816_XXD0yTNei50_id01002_wavtolip.mp4"

if not os.path.exists(video_path):
    print(f"File {video_path} not found.")
else:
    with open(video_path, "rb") as f:
        files = {"file": (os.path.basename(video_path), f, "video/mp4")}
        try:
            print(f"Sending request to {url}...")
            response = requests.post(url, files=files)
            print(f"Status Code: {response.status_code}")
            if response.status_code == 200:
                print(f"Response: {response.json()}")
            else:
                print(f"Error Response: {response.text}")
        except Exception as e:
            print(f"Error: {e}")
