import math

import numpy as np


def _validate_inputs(y_true, scores):
    """Validate and normalize labels and scores shared by the metrics."""
    try:
        y = np.asarray(y_true)
        s = np.asarray(scores, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Inputs must contain valid labels and numeric scores") from exc

    if y.ndim != 1 or s.ndim != 1:
        raise ValueError("Inputs must be one-dimensional")

    if len(y) != len(s):
        raise ValueError("y_true and scores must have equal lengths")

    if len(y) == 0:
        raise ValueError("Inputs must not be empty")

    if not np.isin(y, [0, 1]).all():
        raise ValueError("Labels must be 0 or 1")

    if np.isnan(s).any():
        raise ValueError("Scores must not contain NaN")

    return y, s


def confusion_at(y_true, scores, threshold):
    """Return TP, FP, TN, and FN at the given threshold."""
    y, s = _validate_inputs(y_true, scores)

    predicted = (s >= threshold)

    tp = np.sum((y == 1) & predicted)
    fp = np.sum((y == 0) & predicted)
    tn = np.sum((y == 0) & ~predicted)
    fn = np.sum((y == 1) & ~predicted)

    return {
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def precision_recall_at(y_true, scores, threshold):
    """Return (precision, recall) at the given threshold."""
    counts = confusion_at(y_true, scores, threshold)

    tp = counts["tp"]
    fp = counts["fp"]
    fn = counts["fn"]

    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0

    return float(precision), float(recall)


def precision_at_k(y_true, scores, k):
    """Return the fraction of positives among the top k scores.

    Ties preserve their original order. If k exceeds the number of
    observations, all observations are used.
    """
    y, s = _validate_inputs(y_true, scores)

    if isinstance(k, (bool, np.bool_)) or not isinstance(
        k, (int, np.integer)
    ):
        raise ValueError("k must be a positive integer")

    if k <= 0:
        raise ValueError("k must be positive")

    k = min(int(k), len(y))

    # Stable sorting preserves original order when scores are tied.
    order = np.argsort(-s, kind="stable")
    top_k_labels = y[order[:k]]

    return float(np.mean(top_k_labels == 1))


def expected_cost(y_true, scores, threshold, *, cost_fp, cost_fn):
    """Calculate the total cost of false positives and false negatives."""
    counts = confusion_at(y_true, scores, threshold)

    return float(
        counts["fp"] * cost_fp
        + counts["fn"] * cost_fn
    )


def best_threshold(y_true, scores, *, cost_fp, cost_fn):
    """Return the threshold with the lowest expected cost.

    Candidates are every distinct score plus math.inf. If multiple
    thresholds have the same minimum cost, choose the highest one.
    """
    _, s = _validate_inputs(y_true, scores)

    candidates = sorted(set(s.tolist()))
    candidates.append(math.inf)

    best_t = None
    lowest_cost = math.inf

    for threshold in candidates:
        cost = expected_cost(
            y_true,
            s,
            threshold,
            cost_fp=cost_fp,
            cost_fn=cost_fn,
        )

        if cost < lowest_cost or (
            cost == lowest_cost
            and (best_t is None or threshold > best_t)
        ):
            lowest_cost = cost
            best_t = threshold

    return float(best_t)


def threshold_for_precision(y_true, scores, min_precision):
    """Return the lowest distinct-score threshold meeting min_precision.

    Return None if no candidate threshold reaches the required precision.
    """
    _, s = _validate_inputs(y_true, scores)

    candidates = sorted(set(s.tolist()))
    valid_thresholds = []

    for threshold in candidates:
        precision, _ = precision_recall_at(
            y_true,
            s,
            threshold,
        )

        if precision >= min_precision:
            valid_thresholds.append(threshold)

    if not valid_thresholds:
        return None

    return float(min(valid_thresholds))
