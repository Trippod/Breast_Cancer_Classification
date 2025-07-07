# src/models/model_logistic.py
from pathlib import Path
import joblib, seaborn as sns, matplotlib.pyplot as plt, pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

from data_preprocessing import prepare_data

# ── katalogi ───────────────────────────────────────────────
ROOT   = Path(__file__).resolve().parents[1]             # …/venv
PKLDIR = ROOT / "models"
RESDIR = ROOT / "outputs" / "models_res" / "logistic"
PKLDIR.mkdir(exist_ok=True)
RESDIR.mkdir(parents=True, exist_ok=True)

# ── główna funkcja treningowa ──────────────────────────────
def train_logistic() -> None:
    X_tr, X_te, y_tr, y_te = prepare_data()

    clf = LogisticRegression(
        max_iter=5000,
        solver="liblinear",
        class_weight="balanced",
        n_jobs=1,
        random_state=42,
    )
    clf.fit(X_tr, y_tr)
    joblib.dump(clf, PKLDIR / "logistic.pkl")

    y_pred  = clf.predict(X_te)
    auc     = roc_auc_score(y_te, clf.predict_proba(X_te)[:, 1])

    _report_and_plot(
        title="Logistic Regression",
        y_true=y_te,
        y_pred=y_pred,
        auc=auc,
        save_path=RESDIR / "confusion_matrix.png",
    )
    print(f"✓ model  → {PKLDIR / 'logistic.pkl'}")

# ── raport + heat-mapa ─────────────────────────────────────
def _report_and_plot(title: str,
                     y_true, y_pred,
                     auc: float,
                     save_path: Path) -> None:
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec  = recall_score(y_true, y_pred)
    f1   = f1_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    spec = tn / (tn + fp)

    print(f"\n=== {title} ===")
    print(f"Accuracy     : {acc:.3f}")
    print(f"Precision    : {prec:.3f}")
    print(f"Recall       : {rec:.3f}")
    print(f"Specificity  : {spec:.3f}")
    print(f"F1-score     : {f1:.3f}")
    print(f"AUC-ROC      : {auc:.3f}")
    print(classification_report(y_true, y_pred,
                                target_names=["Benign", "Malignant"]))

    cm_df = pd.DataFrame(
        [[tn, fp], [fn, tp]],
        index=["Benign (0)", "Malignant (1)"],
        columns=["Pred 0", "Pred 1"],
    )
    sns.set(style="whitegrid", palette="muted")
    plt.figure(figsize=(4, 4))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f"Confusion Matrix – {title}")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.clf()
    print(f"✓ heatmap → {save_path}")

# ── uruchomienie modułu ────────────────────────────────────
if __name__ == "__main__":
    train_logistic()
