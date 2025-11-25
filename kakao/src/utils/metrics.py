import numpy as np
# ======== Weighted R² ========
def weighted_r2_score(y_true: np.ndarray, y_pred: np.ndarray):
    """
    y_true, y_pred: shape (N, 5)
    """
    weights = np.array([0.1, 0.1, 0.1, 0.2, 0.5])
    r2_scores = []
    for i in range(5):
        y_t = y_true[:, i]
        y_p = y_pred[:, i]
        ss_res = np.sum((y_t - y_p) ** 2)
        ss_tot = np.sum((y_t - np.mean(y_t)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        r2_scores.append(r2)
    r2_scores = np.array(r2_scores)
    weighted_r2 = np.sum(r2_scores * weights) / np.sum(weights)
    return weighted_r2, r2_scores