"""
01_install_requirements.py
Run this first, in a Colab cell (use ! or %pip) or as a script locally.

Colab cell version:
    !pip -q install mediapipe==0.10.14 opencv-python-headless scikit-learn pandas tqdm

Windows local machine (for the webcam demo, stage 14):
    pip install mediapipe opencv-python scikit-learn pandas tqdm tensorflow playsound==1.2.2

Notes:
- Colab already ships tensorflow (2.16+, which bundles Keras 3) — no need to reinstall
  unless you need a specific version; if you do:
      !pip -q install "tensorflow>=2.16,<2.18"
- On Windows, use opencv-python (not headless) so cv2.imshow works for the live demo.
- mediapipe wheels are published for Python 3.9-3.11 on Windows; if you're on a newer
  Python, use a 3.10/3.11 venv for the local webcam machine.
- playsound is only needed for the simple cross-platform alarm beep in stage 14;
  on Windows you can alternatively use the built-in `winsound` module (no install needed).
"""

import subprocess
import sys

PACKAGES = [
    "mediapipe==0.10.14",
    "opencv-python-headless",   # swap to "opencv-python" on the local Windows webcam machine
    "scikit-learn",
    "pandas",
    "tqdm",
]

def pip_install(packages):
    for pkg in packages:
        print(f"Installing {pkg} ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

if __name__ == "__main__":
    pip_install(PACKAGES)
    print("Done. TensorFlow is assumed pre-installed (Colab) or install separately:")
    print('  pip install "tensorflow>=2.16,<2.18"')
