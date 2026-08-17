"""Generate real ROC/PR curves from the canonical model's actual DFDC
held-out predictions (outputs/predictions/dfdc_error_analysis.csv). No
synthetic or placeholder data -- these are the same 400 DFDC predictions
(label, prob) that back every number in Table III row 4 / Section V-B."""
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

path = r"D:/deepfake_detection/deepfake_detection_project/outputs/predictions/dfdc_error_analysis.csv"
y_true, y_score = [], []
with open(path, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        y_true.append(1 if row["label"].strip().lower() == "fake" else 0)
        y_score.append(float(row["prob"]))

print(f"n={len(y_true)}  positives(fake)={sum(y_true)}  negatives(real)={len(y_true)-sum(y_true)}")

fpr, tpr, _ = roc_curve(y_true, y_score)
roc_auc = auc(fpr, tpr)
prec, rec, _ = precision_recall_curve(y_true, y_score)
ap = average_precision_score(y_true, y_score)
print(f"ROC AUC = {roc_auc:.4f}  (matches Table III attempt-4 DFDC AUC = 0.7105)")
print(f"Average precision = {ap:.4f}")

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.45, 1.75))

ax1.plot(fpr, tpr, color="#1a5276", linewidth=1.3, label=f"AUC = {roc_auc:.3f}")
ax1.plot([0, 1], [0, 1], color="gray", linewidth=0.8, linestyle="--")
ax1.set_xlabel("False Positive Rate")
ax1.set_ylabel("True Positive Rate")
ax1.set_title("(a) ROC")
ax1.legend(loc="lower right", frameon=False)
ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1.02)

base_rate = sum(y_true) / len(y_true)
ax2.plot(rec, prec, color="#a93226", linewidth=1.3, label=f"AP = {ap:.3f}")
ax2.axhline(base_rate, color="gray", linewidth=0.8, linestyle="--", label=f"base rate = {base_rate:.2f}")
ax2.set_xlabel("Recall")
ax2.set_ylabel("Precision")
ax2.set_title("(b) Precision--Recall")
ax2.legend(loc="lower left", frameon=False)
ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1.02)

for ax in (ax1, ax2):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

fig.tight_layout(pad=0.4, w_pad=1.2)
out = r"D:/deepfake_detection/deepfake_detection_project/paper/paperoverleaf/figures/dfdc_roc_pr.pdf"
fig.savefig(out)
print("saved:", out)
