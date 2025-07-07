from pathlib import Path
import numpy as np, pandas as pd, seaborn as sns, matplotlib.pyplot as plt
from datetime import datetime
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)
from data_preprocessing import prepare_data

ROOT   = Path(__file__).resolve().parents[1]          # …/venv
PKLDIR = ROOT / "models"
RESDIR = ROOT / "outputs" / "models_res" / "nn"
PKLDIR.mkdir(exist_ok=True)
RESDIR.mkdir(parents=True, exist_ok=True)

# ───────────── 1. Budujemy i trenujemy ─────────────
def build_model(input_dim: int) -> tf.keras.Model:
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid")
    ])
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
    )
    return model

def train_nn() -> None:
    X_tr, X_te, y_tr, y_te = prepare_data()
    model = build_model(X_tr.shape[1])

    early = callbacks.EarlyStopping(
        patience=15, restore_best_weights=True, monitor="val_auc"
    )
    hist = model.fit(
        X_tr, y_tr,
        validation_split=0.2,
        epochs=50,
        batch_size=32,
        callbacks=[early],
        verbose=0
    )

    model.save(PKLDIR / "nn.h5")

    # ─── Ewaluacja ───
    y_pred_prob = model.predict(X_te).ravel()
    y_pred = (y_pred_prob >= 0.5).astype(int)

    report_and_plot(
        title="Neural Network",
        y_true=y_te,
        y_pred=y_pred,
        auc=roc_auc_score(y_te, y_pred_prob),
        save_path=RESDIR / "confusion_matrix.png"
    )
    save_metrics(hist, y_te, y_pred_prob)

# ───────────── 2. Raport + heat-mapa ─────────────
def report_and_plot(title, y_true, y_pred, auc, save_path):
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec  = recall_score(y_true, y_pred)
    f1   = f1_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    spec = tn / (tn + fp)

    print(f"\n=== {title} ===")
    for k, v in [("Accuracy",acc),("Precision",prec),("Recall",rec),
                 ("Specificity",spec),("F1",f1),("AUC-ROC",auc)]:
        print(f"{k:12s}: {v:.3f}")
    print(classification_report(y_true, y_pred,
                                target_names=["Benign","Malignant"]))

    cm_df = pd.DataFrame([[tn, fp], [fn, tp]],
                         index=["Benign(0)","Malignant(1)"],
                         columns=["Pred 0","Pred 1"])
    sns.set(style="whitegrid", palette="muted")
    plt.figure(figsize=(4,4))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f"Confusion Matrix – {title}")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.clf()
    print(f"✓ heatmap → {save_path}")

# ───────────── 3. Zapis CSV z metrykami ─────────────
def save_metrics(hist, y_true, y_prob):
    # ostatnia epoka val_...
    auc = roc_auc_score(y_true, y_prob)
    data = {
        "final_val_accuracy": hist.history["val_accuracy"][-1],
        "final_val_auc"     : hist.history["val_auc"][-1],
        "test_auc"          : auc,
        "timestamp"         : datetime.now()
    }
    pd.DataFrame([data]).to_csv(RESDIR / "metrics.csv", index=False)
    print(f"✓ metrics → {RESDIR/'metrics.csv'}")

# ───────────── main ─────────────
if __name__ == "__main__":
    train_nn()
