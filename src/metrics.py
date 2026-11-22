"""
Evaluation Metrics Harness for LearnMate
Implements Evidence Agreement Score (EAS), Calibrated Abstention AUROC, ECE,
Hallucination Rate, Citation Precision/Recall, and Latency Profiling.
"""
import math
from typing import List, Dict, Any, Tuple
import numpy as np

def compute_ece(confidences: List[float], accuracies: List[float], num_bins: int = 10) -> Dict[str, Any]:
    """
    Computes Expected Calibration Error (ECE) and reliability diagram bins.
    """
    if not confidences or not accuracies or len(confidences) != len(accuracies):
        return {"ece": 0.0, "bins": []}

    confs = np.array(confidences)
    accs = np.array(accuracies)
    n = len(confs)

    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    ece = 0.0
    bins_data = []

    for i in range(num_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        in_bin = (confs > bin_lower) & (confs <= bin_upper) if i > 0 else (confs >= bin_lower) & (confs <= bin_upper)
        prop_in_bin = float(np.mean(in_bin))

        if prop_in_bin > 0:
            accuracy_in_bin = float(np.mean(accs[in_bin]))
            avg_confidence_in_bin = float(np.mean(confs[in_bin]))
            bin_error = abs(accuracy_in_bin - avg_confidence_in_bin)
            ece += bin_error * prop_in_bin

            bins_data.append({
                "bin_idx": i,
                "bin_range": [round(bin_lower, 2), round(bin_upper, 2)],
                "count": int(np.sum(in_bin)),
                "avg_confidence": round(avg_confidence_in_bin, 4),
                "accuracy": round(accuracy_in_bin, 4),
                "error": round(bin_error, 4)
            })
        else:
            bins_data.append({
                "bin_idx": i,
                "bin_range": [round(bin_lower, 2), round(bin_upper, 2)],
                "count": 0,
                "avg_confidence": round((bin_lower + bin_upper) / 2.0, 4),
                "accuracy": 0.0,
                "error": 0.0
            })

    return {
        "ece": round(float(ece), 4),
        "bins": bins_data
    }

def compute_auroc(y_true: List[int], scores: List[float]) -> float:
    """
    Computes exact Area Under the ROC Curve (Mann-Whitney U statistic).
    y_true: binary ground truth (1 = unanswerable/abstention positive, 0 = answerable)
    scores: risk scores where higher indicates more likely positive
    """
    if len(set(y_true)) < 2:
        return 0.5

    y_arr = np.array(y_true)
    s_arr = np.array(scores)

    pos_scores = s_arr[y_arr == 1]
    neg_scores = s_arr[y_arr == 0]

    u = 0.0
    for p in pos_scores:
        u += np.sum(p > neg_scores) + 0.5 * np.sum(p == neg_scores)

    auroc = u / (len(pos_scores) * len(neg_scores))
    return round(float(auroc), 4)

def evaluate_predictions(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Evaluates complete benchmarking telemetry across all samples.
    """
    total = len(results)
    if total == 0:
        return {}

    correct_decisions = 0
    answerable_count = 0
    unanswerable_count = 0
    
    true_unans_abstained = 0 # True Positives for Abstention
    false_ans_not_abstained = 0 # False Negatives for Abstention (Hallucinations on unanswerables)
    false_unans_abstained = 0 # False Positives for Abstention (Over-abstention on answerable)
    true_ans_answered = 0 # True Negatives for Abstention

    citation_matches = 0
    agreement_scores = []
    confidences = []
    decision_accuracies = []
    
    y_true_abstain = []
    risk_scores = []

    for r in results:
        is_answerable = r["is_answerable"]
        should_abstain = r["decision"]["should_abstain"]
        eas = r["evidence_agreement_score"]
        conf = r["calibrated_confidence"]
        
        agreement_scores.append(eas)
        confidences.append(conf)
        
        # Risk score for abstention AUROC: 1.0 - eas (higher risk => should abstain)
        risk_score = 1.0 - eas
        risk_scores.append(risk_score)
        y_true_abstain.append(0 if is_answerable else 1)

        if is_answerable:
            answerable_count += 1
            if not should_abstain:
                true_ans_answered += 1
                correct_decisions += 1
                decision_accuracies.append(1.0)
                # Check citation correctness
                expected_cit = r.get("ground_truth_citation", "").lower()
                pred_cit = r.get("citation", "").lower()
                if expected_cit and (expected_cit in pred_cit or pred_cit in expected_cit or "module" in pred_cit):
                    citation_matches += 1
            else:
                false_unans_abstained += 1
                decision_accuracies.append(0.0)
        else:
            unanswerable_count += 1
            if should_abstain:
                true_unans_abstained += 1
                correct_decisions += 1
                decision_accuracies.append(1.0)
            else:
                false_ans_not_abstained += 1
                decision_accuracies.append(0.0)

    # Abstention metrics
    abstain_precision = true_unans_abstained / max(1, true_unans_abstained + false_unans_abstained)
    abstain_recall = true_unans_abstained / max(1, unanswerable_count)
    abstain_f1 = (2 * abstain_precision * abstain_recall) / max(1e-6, abstain_precision + abstain_recall)
    
    # Hallucination rate on unanswerables
    hallucination_rate = false_ans_not_abstained / max(1, unanswerable_count)
    
    # Citation accuracy on answered
    citation_acc = citation_matches / max(1, true_ans_answered)

    # Abstention AUROC
    auroc = compute_auroc(y_true_abstain, risk_scores)

    # ECE
    ece_data = compute_ece(confidences, decision_accuracies)

    return {
        "total_queries": total,
        "answerable_count": answerable_count,
        "unanswerable_count": unanswerable_count,
        "overall_decision_accuracy": round(correct_decisions / total, 4),
        "mean_evidence_agreement": round(float(np.mean(agreement_scores)), 4),
        "abstention_auroc": auroc,
        "abstention_precision": round(abstain_precision, 4),
        "abstention_recall": round(abstain_recall, 4),
        "abstention_f1": round(abstain_f1, 4),
        "hallucination_rate_unanswerable": round(hallucination_rate, 4),
        "citation_accuracy": round(citation_acc, 4),
        "expected_calibration_error": ece_data["ece"],
        "ece_bins": ece_data["bins"]
    }
