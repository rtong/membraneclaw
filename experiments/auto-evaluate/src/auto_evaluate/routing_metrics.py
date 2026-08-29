from __future__ import annotations

import math
from typing import Any, Iterable


ROUTE_ACTIONS = {"skip_rag", "use_rag"}


def binary_route_metrics(
    rows: Iterable[dict[str, Any]],
    *,
    expected_key: str,
    predicted_key: str,
) -> dict[str, Any]:
    """Summarize use_rag as the positive class for a mixed routing set."""
    labeled = [
        row
        for row in rows
        if row.get(expected_key) in ROUTE_ACTIONS
        and row.get(predicted_key) in ROUTE_ACTIONS
    ]
    tp = sum(
        row[expected_key] == "use_rag" and row[predicted_key] == "use_rag"
        for row in labeled
    )
    tn = sum(
        row[expected_key] == "skip_rag" and row[predicted_key] == "skip_rag"
        for row in labeled
    )
    fp = sum(
        row[expected_key] == "skip_rag" and row[predicted_key] == "use_rag"
        for row in labeled
    )
    fn = sum(
        row[expected_key] == "use_rag" and row[predicted_key] == "skip_rag"
        for row in labeled
    )

    def ratio(numerator: int, denominator: int) -> float | None:
        return numerator / denominator if denominator else None

    precision = ratio(tp, tp + fp)
    recall = ratio(tp, tp + fn)
    specificity = ratio(tn, tn + fp)
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else None
    )
    balanced_accuracy = (
        (recall + specificity) / 2
        if recall is not None and specificity is not None
        else None
    )
    return {
        "n_labeled": len(labeled),
        "support": {
            "skip_rag": tn + fp,
            "use_rag": tp + fn,
        },
        "confusion_matrix": {
            "true_skip_pred_skip": tn,
            "true_skip_pred_use": fp,
            "true_use_pred_skip": fn,
            "true_use_pred_use": tp,
        },
        "accuracy": ratio(tp + tn, len(labeled)),
        "use_rag_precision": precision,
        "use_rag_recall": recall,
        "use_rag_f1": f1,
        "skip_rag_specificity": specificity,
        "balanced_accuracy": balanced_accuracy,
        "false_positive_rate": ratio(fp, fp + tn),
        "false_negative_rate": ratio(fn, fn + tp),
    }


def paired_exact_mcnemar(
    baseline_rows: Iterable[dict[str, Any]],
    candidate_rows: Iterable[dict[str, Any]],
    *,
    identity_key: str = "case_id",
    correct_key: str = "routing_correct",
) -> dict[str, Any]:
    """Return paired discordance counts and an exact two-sided McNemar p-value."""
    baseline = {
        row.get(identity_key): row.get(correct_key)
        for row in baseline_rows
        if row.get(identity_key) is not None and isinstance(row.get(correct_key), bool)
    }
    candidate = {
        row.get(identity_key): row.get(correct_key)
        for row in candidate_rows
        if row.get(identity_key) is not None and isinstance(row.get(correct_key), bool)
    }
    identities = sorted(set(baseline) & set(candidate), key=str)
    baseline_wrong_candidate_right = sum(
        baseline[key] is False and candidate[key] is True for key in identities
    )
    baseline_right_candidate_wrong = sum(
        baseline[key] is True and candidate[key] is False for key in identities
    )
    discordant = baseline_wrong_candidate_right + baseline_right_candidate_wrong
    p_value = None
    if discordant:
        lower = min(
            baseline_wrong_candidate_right,
            baseline_right_candidate_wrong,
        )
        tail = sum(math.comb(discordant, index) for index in range(lower + 1))
        p_value = min(1.0, 2 * tail / (2**discordant))
    return {
        "n_paired": len(identities),
        "baseline_wrong_candidate_right": baseline_wrong_candidate_right,
        "baseline_right_candidate_wrong": baseline_right_candidate_wrong,
        "n_discordant": discordant,
        "exact_two_sided_p": p_value,
    }
