import subprocess
import sys

PIPELINE_STEPS = [
    "pipeline/phase1_organize.py",
    "pipeline/phase2_clean.py",
    "pipeline/phase3_normalize.py",
    "pipeline/phase4_split.py",
    "pipeline/phase5_extract.py",
]

def run_step(script):
    print(f"Running {script}")
    subprocess.run([sys.executable, script], check=True)

def main():
    for step in PIPELINE_STEPS:
        run_step(step)
    print("Pipeline completed successfully")

if __name__ == "__main__":
    main()
