import numpy as np
try:
    from sklearn.metrics import roc_auc_score
    _HAS_SK = True
except Exception:
    _HAS_SK = False


def compute_metrics(outputs_sig, targets, mask, threshold=0.5):
    # outputs_sig, targets, mask are same shape; values in {0,1} for targets/mask
    probs = (outputs_sig * mask).flatten()
    labels = (targets * mask).flatten()

    preds = (probs >= threshold).float()
    tp = (preds * labels).sum()
    fp = (preds * (1 - labels)).sum()
    fn = ((1 - preds) * labels).sum()
    tn = (((1 - preds) * (1 - labels))).sum()

    tp, fp, fn, tn = [x.item() for x in (tp, fp, fn, tn)]
    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)
    acc       = (tp + tn) / (tp + tn + fp + fn + 1e-8)

    roc_auc = None
    if _HAS_SK:
        # Only feed valid (mask==1) positions to ROC-AUC
        m = mask.flatten().bool()
        y_true = labels[m].detach().cpu().numpy()
        y_prob = probs[m].detach().cpu().numpy()
        if y_true.size > 0 and len(np.unique(y_true)) > 1:
            roc_auc = float(roc_auc_score(y_true, y_prob))

    return {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": roc_auc
    }
