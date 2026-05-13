"""
Download the Kaggle Chest X-Ray dataset automatically.

Prerequisites:
    1. Create a free Kaggle account: https://www.kaggle.com
    2. Go to Account → API → Create New Token  (downloads kaggle.json)
    3. Place kaggle.json in ~/.kaggle/kaggle.json
       Linux/Mac:  cp ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
                   chmod 600 ~/.kaggle/kaggle.json
       Windows:    %USERPROFILE%\.kaggle\kaggle.json

Then run:
    python download_data.py
"""
import os
import subprocess
import sys
import zipfile


DATASET   = "paultimothymooney/chest-xray-pneumonia"
DATA_DIR  = "data"
ZIP_NAME  = "chest-xray-pneumonia.zip"


def check_kaggle():
    try:
        import kaggle  # noqa: F401
    except ImportError:
        print("Installing kaggle package…")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "kaggle", "-q"])


def download():
    check_kaggle()
    os.makedirs(DATA_DIR, exist_ok=True)

    print(f"Downloading '{DATASET}' from Kaggle…")
    subprocess.check_call([
        sys.executable, "-m", "kaggle", "datasets", "download",
        "-d", DATASET,
        "-p", DATA_DIR,
    ])

    zip_path = os.path.join(DATA_DIR, ZIP_NAME)
    print(f"Extracting {zip_path}…")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(DATA_DIR)
    os.remove(zip_path)
    print(f"Dataset ready at {DATA_DIR}/chest_xray/")


if __name__ == "__main__":
    download()
