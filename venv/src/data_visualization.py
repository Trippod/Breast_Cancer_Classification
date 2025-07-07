# src/data_visualization.py
import time
from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from data_preprocessing import load_data

# ────────── styl globalny ──────────
sns.set(style="whitegrid", palette="muted")
OUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "eda"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def save(fig_path: Path | None) -> None:
    '''

    :param fig_path:
    :return:
    '''
    if fig_path:
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(fig_path)
        plt.clf()
        print(f"✓ zapisano {fig_path.name}")
    else:
        plt.show()


def _std_long(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    '''

    :param df:
    :param cols:
    :return:
    '''
    std_part = (df[cols] - df[cols].mean()) / df[cols].std()
    return (
        pd.concat([df["diagnosis"], std_part], axis=1)
        .melt(id_vars="diagnosis", var_name="features", value_name="value")
    )


# ────────── 1. Violin & Strip (po 10 cech) ──────────
def plot_group(df: pd.DataFrame, idx_from: int, idx_to: int) -> None:
    '''

    :param df:
    :param idx_from:
    :param idx_to:
    :return:
    '''
    features_all = [
        c for c in df.columns if c not in ("id", "diagnosis", "Unnamed: 32")
    ]
    cols = features_all[idx_from:idx_to]
    data_long = _std_long(df, cols)

    # Violin
    plt.figure(figsize=(10, 6))
    sns.violinplot(
        x="features", y="value",
        hue="diagnosis", data=data_long,
        split=True, inner="quart"
    )
    plt.xticks(rotation=90)
    save(OUT_DIR / f"violin_{idx_from}_{idx_to-1}.png")

    # Strip
    plt.figure(figsize=(10, 6))
    tic = time.time()
    sns.stripplot(
        x="features", y="value",
        hue="diagnosis", data=data_long,
        dodge=True, jitter=True,
        size=4, alpha=.6
    )
    plt.xticks(rotation=90)
    print(f"  stripplot {idx_from}-{idx_to-1} rysował się {time.time()-tic:.1f}s")
    save(OUT_DIR / f"strip_{idx_from}_{idx_to-1}.png")


# ────────── 2. Heatmap korelacji (wszystkie 30 cech) ──────────
def plot_corr_heatmap(df: pd.DataFrame, save_path: Path | None = None) -> None:
    '''

    :param df:
    :param save_path:
    :return:
    '''
    features = df.drop(columns=["id", "diagnosis", "Unnamed: 32"], errors="ignore")

    plt.figure(figsize=(18, 18))
    sns.heatmap(
        features.corr(),
        annot=True, linewidths=.5, fmt=".2f",
        cmap="coolwarm", square=True
    )
    plt.title("Macierz korelacji – 30 cech")
    save(save_path)



if __name__ == "__main__":
    df_raw = load_data()
    df_raw["diagnosis"] = df_raw["diagnosis"].map({"M": 1, "B": 0})

    # trzy paczki po 10 cech
    for start in range(0, 30, 10):
        plot_group(df_raw, start, start + 10)

    # pełna heatmapa
    plot_corr_heatmap(df_raw, OUT_DIR / "corr_heatmap.png")
