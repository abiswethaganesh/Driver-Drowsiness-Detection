"""
00_config.py
Shared configuration for the whole NITYMED drowsiness pipeline.
Import this from every other stage:  from config import CFG
"""

import os

class CFG:
    # ---------------- Paths ----------------
    # Root folder where the NITYMED dataset is located locally
    NITYMED_ROOT = r"D:\Sem 7\NNDL\Project Demo\DSM_Dataset-HDTV720"

    WORK_DIR = r"D:\Sem 7\NNDL\Project Demo\nitymed_work"           # all derived artifacts go here
    METADATA_CSV = os.path.join(WORK_DIR, "metadata.csv")
    SPLIT_CSV = os.path.join(WORK_DIR, "split.csv")
    FRAME_METADATA_CSV = os.path.join(WORK_DIR, "frame_metadata.csv")
    FACE_LANDMARK_METADATA_CSV = os.path.join(WORK_DIR, "face_landmark_metadata.csv")
    FRAMES_DIR = os.path.join(WORK_DIR, "frames")            # stage 5 output
    FACES_DIR = os.path.join(WORK_DIR, "faces")              # stage 6 output face crop JPEGs
    LANDMARKS_DIR = os.path.join(WORK_DIR, "landmarks")      # stage 6 output per-video .npz arrays
    SEQ_DIR = os.path.join(WORK_DIR, "sequences")             # stage 7 output (train/val/test .npz)
    MODEL_DIR = os.path.join(WORK_DIR, "models")
    CALIB_DIR = os.path.join(WORK_DIR, "driver_calibration")

    # ---------------- Video sampling ----------------
    NATIVE_FPS = 25            # confirmed: NITYMED is recorded at 25 fps
    FRAME_STRIDE = 3           # keep every 3rd frame -> ~8.33 effective fps
    IMG_SIZE = 128             # face crop resized to IMG_SIZE x IMG_SIZE
    SEQ_LEN = 16               # frames per sequence -> 16*3/25 = 1.92s temporal window
    SEQ_OVERLAP = 0.5          # 50% overlap between consecutive training sequences

    # ---------------- Driver-id parsing (ADJUST after running stage 02) ----------------
    # Default assumes filenames like "D07_Yawn_2.mp4" or "07_microsleep_1.mp4"
    DRIVER_ID_REGEX = r"(?:^|[_\-\/])([Dd]?\d{1,3})(?:[_\-])"

    # ---------------- Category names as they appear in folder/file names ----------------
    YAWN_KEYWORDS = ["yawn", "yawning"]
    MICROSLEEP_KEYWORDS = ["microsleep", "micro-sleep", "micro_sleep"]

    # ---------------- Pseudo-labeling thresholds (EAR / MAR heuristics) ----------------
    # EAR: lower = eyes more closed. MAR: higher = mouth more open (yawn-like).
    EAR_CLOSED_THRESH = 0.20          # below this counts as "closed"
    EAR_CLOSED_MIN_FRAMES = 8         # sustained closure (at effective ~8.3fps -> ~1s) => microsleep-like
    MAR_YAWN_THRESH = 0.55            # above this counts as "wide open" (yawn candidate)
    MAR_YAWN_MIN_FRAMES = 5           # sustained wide-mouth -> ~0.6s at effective fps

    # fraction of positive (event) frames inside a sequence window required to call
    # the WHOLE sequence "drowsiness-indicative"
    SEQ_POSITIVE_FRAME_RATIO = 0.35

    # ---------------- Splits ----------------
    VAL_FRACTION = 0.15
    TEST_FRACTION = 0.15
    RANDOM_SEED = 42

    # ---------------- Model / training ----------------
    CNN_BACKBONE = "MobileNetV2"      # or "EfficientNetB0"
    FEATURE_DIM = 1280                # MobileNetV2 GAP output dim
    LSTM_UNITS = 64
    DROPOUT = 0.4
    BATCH_SIZE = 16
    EPOCHS = 25
    LEARNING_RATE = 1e-3
    FINE_TUNE_LEARNING_RATE = 1e-5
    FINE_TUNE_UNFREEZE_LAYERS = 25    # how many trailing CNN layers to unfreeze if fine-tuning

    # ---------------- Decision thresholds ----------------
    GLOBAL_THRESHOLD = 0.5            # baseline system decision threshold
    PERSONALIZED_THRESHOLD = 0.5      # applied to the *personalized* combined risk score

    # ---------------- Personal calibration ----------------
    CALIBRATION_SECONDS = 12          # 10-15s window, pick a concrete value
    CALIBRATION_MIN_SEQUENCES = 6     # minimum sequences needed to trust a baseline
    BASELINE_EMA_ALPHA = 0.02         # slow-update rate for the running baseline
    BASELINE_UPDATE_RISK_CEILING = 0.3  # only update baseline from sequences below this risk
    DEVIATION_WEIGHT = 1.5            # weight of personal deviation term in logit combination

    @classmethod
    def ensure_dirs(cls):
        for d in [cls.WORK_DIR, cls.FRAMES_DIR, cls.FACES_DIR, cls.LANDMARKS_DIR,
                  cls.SEQ_DIR, cls.MODEL_DIR, cls.CALIB_DIR]:
            os.makedirs(d, exist_ok=True)
