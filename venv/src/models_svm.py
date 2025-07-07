from pathlib import Path
import joblib, seaborn as sns, matplotlib.pyplot as plt, pandas as pd

from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from data_preprocessing import prepare_data

ROOT   = Path(__file__).resolve().parents[1]          # …/venv
PKLDIR = ROOT / "models"
RESDIR = ROOT / "outputs" / "models_res" / "svm"
PKLDIR.mkdir(exist_ok=True)
RESDIR.mkdir(parents=True, exist_ok=True)

def train_svm() -> None:
    X_tr, X_te, y_tr, y_te = prepare_data()

    clf = SVC(
        kernel="rbf",
        C=10,
        gamma="scale",
        probability=True,        # potrzebne do AUC
        class_weight="balanced",
        random_state=42,
    )
    clf.fit(X_tr, y_tr)
    joblib.dump(clf, PKLDIR / "svm.pkl")

    y_pred  = clf.predict(X_te)
    y_proba = clf.predict_proba(X_te)[:, 1]
    auc     = roc_auc_score(y_te, y_proba)

    _report_and_plot("SVM (RBF)", y_te, y_pred, auc,
                     RESDIR / "confusion_matrix.png")
    print(f"✓ model  → {PKLDIR/'svm.pkl'}")

# ───── pomocnicze raportowanie + heat-mapa ─────
def _report_and_plot(title, y_true, y_pred, auc, save_path):
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec  = recall_score(y_true, y_pred)
    f1   = f1_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    spec = tn / (tn + fp)

    print(f"\n=== {title} ===")
    for k, v in [("Accuracy", acc), ("Precision", prec), ("Recall", rec),
                 ("Specificity", spec), ("F1-score", f1), ("AUC-ROC", auc)]:
        print(f"{k:12s}: {v:.3f}")
    print(classification_report(y_true, y_pred,
                                target_names=["Benign","Malignant"]))

    cm_df = pd.DataFrame([[tn, fp], [fn, tp]],
                         index=["Benign(0)", "Malignant(1)"],
                         columns=["Pred 0", "Pred 1"])
    sns.set(style="whitegrid", palette="muted")
    plt.figure(figsize=(4,4))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f"Confusion Matrix – {title}")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.clf()
    print(f"✓ heatmap → {save_path}")

if __name__ == "__main__":
    train_svm()
