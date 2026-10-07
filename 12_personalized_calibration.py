"""
12_personalized_calibration.py

THE NOVEL CONTRIBUTION: Per-Driver Adaptive Calibration.

Design (see project README section E/F):
  1. CALIBRATION: collect ~10-15s of the driver's own normal/alert behavior
     (CFG.CALIBRATION_SECONDS). Record, per sequence: the CNN-LSTM's raw
     probability, and the mean EAR/MAR over that sequence's frames.
     Fit a personal baseline = (mean, std) for each of these three signals.

  2. RUNTIME: for every new sequence, compute a DEVIATION score of the
     current EAR/MAR against THIS driver's own baseline (not a population
     average), turn it into a z-score, and COMBINE it with the raw CNN-LSTM
     probability in *logit space* -- this nudges the model's own probability
     up or down based on personal deviation, rather than throwing the model
     probability away and hand-thresholding EAR/MAR directly. The CNN-LSTM
     remains the primary signal; personalization is an additive correction.

  3. SLOW BASELINE UPDATE: the personal baseline is allowed to drift slowly
     (EMA, alpha=CFG.BASELINE_EMA_ALPHA) to track gradual changes (e.g.
     lighting, posture) but ONLY using sequences the system currently judges
     low-risk (personalized_prob < CFG.BASELINE_UPDATE_RISK_CEILING). This
     stops a driver who is actually becoming drowsy from dragging their own
     "normal" baseline toward a drowsy state.
"""

import numpy as np
from importlib import import_module
CFG = import_module("00_config").CFG

EPS = 1e-6


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class PersonalCalibrator:
    def __init__(self, driver_id=None):
        self.driver_id = driver_id
        self.is_calibrated = False
        self._calib_probs = []
        self._calib_ears = []
        self._calib_mars = []

        self.baseline_prob_mean = None
        self.baseline_ear_mean = None
        self.baseline_ear_std = None
        self.baseline_mar_mean = None
        self.baseline_mar_std = None

    # ---------------- Calibration phase ----------------
    def add_calibration_sample(self, raw_prob, mean_ear, mean_mar):
        """Call this once per sequence during the 10-15s calibration window."""
        self._calib_probs.append(raw_prob)
        self._calib_ears.append(mean_ear)
        self._calib_mars.append(mean_mar)

    def finalize_calibration(self):
        n = len(self._calib_probs)
        if n < CFG.CALIBRATION_MIN_SEQUENCES:
            raise RuntimeError(
                f"Only {n} calibration sequences collected, need at least "
                f"{CFG.CALIBRATION_MIN_SEQUENCES}. Extend the calibration window."
            )
        self.baseline_prob_mean = float(np.mean(self._calib_probs))
        self.baseline_ear_mean = float(np.mean(self._calib_ears))
        self.baseline_ear_std = float(np.std(self._calib_ears) + EPS)
        self.baseline_mar_mean = float(np.mean(self._calib_mars))
        self.baseline_mar_std = float(np.std(self._calib_mars) + EPS)
        self.is_calibrated = True
        return {
            "driver_id": self.driver_id,
            "baseline_prob_mean": self.baseline_prob_mean,
            "baseline_ear_mean": self.baseline_ear_mean,
            "baseline_ear_std": self.baseline_ear_std,
            "baseline_mar_mean": self.baseline_mar_mean,
            "baseline_mar_std": self.baseline_mar_std,
        }

    def load_baseline(self, baseline_dict):
        self.driver_id = baseline_dict.get("driver_id", self.driver_id)
        self.baseline_prob_mean = baseline_dict["baseline_prob_mean"]
        self.baseline_ear_mean = baseline_dict["baseline_ear_mean"]
        self.baseline_ear_std = baseline_dict["baseline_ear_std"]
        self.baseline_mar_mean = baseline_dict["baseline_mar_mean"]
        self.baseline_mar_std = baseline_dict["baseline_mar_std"]
        self.is_calibrated = True

    # ---------------- Runtime scoring ----------------
    def deviation_score(self, mean_ear, mean_mar):
        """Positive = more drowsy-looking than this driver's own normal."""
        ear_z = (self.baseline_ear_mean - mean_ear) / self.baseline_ear_std   # eyes more closed than usual
        mar_z = (mean_mar - self.baseline_mar_mean) / self.baseline_mar_std   # mouth more open than usual
        dev = (max(ear_z, 0.0) + max(mar_z, 0.0)) / 2.0
        return float(np.clip(dev, -3.0, 3.0))

    def personalized_risk(self, raw_prob, mean_ear, mean_mar):
        if not self.is_calibrated:
            raise RuntimeError("Calibrator not calibrated yet -- call "
                                "finalize_calibration() or load_baseline() first.")
        dev = self.deviation_score(mean_ear, mean_mar)
        combined_logit = logit(raw_prob) + CFG.DEVIATION_WEIGHT * dev
        return float(sigmoid(combined_logit)), dev

    def maybe_update_baseline(self, personalized_prob, mean_ear, mean_mar):
        """Slow EMA update, gated on currently-low-risk sequences only, so a
        driver becoming drowsy cannot drag their own baseline toward 'normal'."""
        if personalized_prob >= CFG.BASELINE_UPDATE_RISK_CEILING:
            return False
        a = CFG.BASELINE_EMA_ALPHA
        self.baseline_ear_mean = (1 - a) * self.baseline_ear_mean + a * mean_ear
        self.baseline_mar_mean = (1 - a) * self.baseline_mar_mean + a * mean_mar
        # widen/narrow std slowly too, floor it so it never collapses to 0
        self.baseline_ear_std = max(EPS, (1 - a) * self.baseline_ear_std +
                                     a * abs(mean_ear - self.baseline_ear_mean))
        self.baseline_mar_std = max(EPS, (1 - a) * self.baseline_mar_std +
                                     a * abs(mean_mar - self.baseline_mar_mean))
        return True

    def decide(self, personalized_prob, threshold=None):
        threshold = threshold if threshold is not None else CFG.PERSONALIZED_THRESHOLD
        return int(personalized_prob >= threshold)
