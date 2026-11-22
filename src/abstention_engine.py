"""
Calibrated Abstention Engine for LearnMate
Implements Conformal Risk Prediction and Temperature-Scaled Confidence Calibration.
"""
import os
import sys
import math
from typing import Dict, Any, List, Tuple
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.config import (
    CONFORMAL_ALPHA, CALIBRATION_TEMPERATURE, DEFAULT_ABSTENTION_THRESHOLD
)

class CalibratedAbstentionEngine:
    """
    Decides whether the AI Tutor should provide a grounded answer or abstain
    based on conformal prediction thresholding and temperature-scaled confidence.
    """
    def __init__(self, 
                 alpha: float = CONFORMAL_ALPHA, 
                 temperature: float = CALIBRATION_TEMPERATURE,
                 default_threshold: float = DEFAULT_ABSTENTION_THRESHOLD):
        self.alpha = alpha
        self.temperature = temperature
        self.calibrated_threshold = default_threshold
        self.is_calibrated = False

    def calibrate(self, val_scores: List[float], val_labels: List[bool]):
        """
        Calibrates the threshold tau* on validation data using split conformal prediction.
        val_scores: list of Evidence Agreement Scores (EAS) in [0, 1]
        val_labels: list of booleans (True if answerable & grounded, False if unanswerable/hallucinated)
        """
        if not val_scores or not val_labels:
            print("Warning: Empty validation data, retaining default threshold.")
            return

        scores = np.array(val_scores)
        labels = np.array(val_labels, dtype=bool)

        # Non-conformity scores: 1.0 - score for grounded answerable items
        grounded_scores = scores[labels]
        if len(grounded_scores) > 0:
            # We want to cover (1 - alpha) of correct answers while rejecting ungrounded ones
            quantile_level = np.clip(self.alpha, 0.01, 0.50)
            tau_star = float(np.quantile(grounded_scores, quantile_level))
            
            # Ensure separation from ungrounded distribution
            ungrounded_scores = scores[~labels]
            if len(ungrounded_scores) > 0:
                max_ungrounded = float(np.max(ungrounded_scores))
                tau_star = max(tau_star, max_ungrounded + 0.02)

            self.calibrated_threshold = min(0.85, max(0.45, tau_star))
            self.is_calibrated = True
            print(f"Calibrated Abstention Threshold tau* set to: {self.calibrated_threshold:.4f} (alpha={self.alpha})")

    def scale_confidence(self, raw_score: float) -> float:
        """
        Applies temperature scaling: p_calib = sigmoid((score - 0.5) / T)
        """
        logit = (raw_score - 0.5) * 4.0
        scaled_logit = logit / max(1e-5, self.temperature)
        prob = 1.0 / (1.0 + math.exp(-scaled_logit))
        return float(prob)

    def evaluate_decision(self, 
                          agreement_score: float, 
                          top_retrieval: Dict[str, Any],
                          verification_details: Dict[str, float]) -> Dict[str, Any]:
        """
        Evaluates whether to answer or abstain.
        Returns detailed decision telemetry.
        """
        conf_scaled = self.scale_confidence(agreement_score)
        contra_score = verification_details.get("contradiction", 0.0)
        sem_sim = verification_details.get("semantic_similarity", 0.0)
        
        # Abstention trigger condition:
        # 1. Agreement score below calibrated threshold
        # 2. Or high contradiction detected
        # 3. Or extremely low retrieval semantic similarity
        should_abstain = (agreement_score < self.calibrated_threshold) or (contra_score >= 0.5) or (sem_sim < 0.20)

        # Categorize refusal reason if abstaining
        if should_abstain:
            if contra_score >= 0.5:
                refusal_type = "CONTRADICTION_DETECTED"
                rationale = "The query premise contradicts established course syllabus definitions."
            elif sem_sim < 0.25:
                refusal_type = "OUT_OF_SYLLABUS"
                rationale = "The topic is outside the approved M.Tech Deep Learning Networks syllabus."
            elif agreement_score < 0.40:
                refusal_type = "INSUFFICIENT_EVIDENCE"
                rationale = "Available course lecture materials lack sufficient verified evidence to support a factual answer."
            else:
                refusal_type = "LOW_CONFIDENCE_CALIBRATED"
                rationale = f"Evidence agreement ({agreement_score:.2f}) falls below calibrated safety threshold ({self.calibrated_threshold:.2f})."
            
            action = "ABSTAIN"
        else:
            refusal_type = "NONE"
            rationale = "Evidence agreement satisfies conformal risk threshold."
            action = "ANSWER"

        return {
            "action": action,
            "should_abstain": should_abstain,
            "calibrated_threshold": self.calibrated_threshold,
            "raw_agreement_score": round(agreement_score, 4),
            "calibrated_confidence": round(conf_scaled, 4),
            "refusal_type": refusal_type,
            "rationale": rationale
        }

if __name__ == "__main__":
    engine = CalibratedAbstentionEngine()
    val_s = [0.72, 0.68, 0.81, 0.75, 0.64, 0.12, 0.05, 0.18, 0.00]
    val_l = [True, True, True, True, True, False, False, False, False]
    engine.calibrate(val_s, val_l)
    
    test_res1 = engine.evaluate_decision(0.70, {}, {"contradiction": 0.0, "semantic_similarity": 0.75})
    test_res2 = engine.evaluate_decision(0.15, {}, {"contradiction": 0.0, "semantic_similarity": 0.10})
    test_res3 = engine.evaluate_decision(0.65, {}, {"contradiction": 0.8, "semantic_similarity": 0.70})
    print("Answerable case:", test_res1)
    print("OOD case:", test_res2)
    print("Contradiction case:", test_res3)
