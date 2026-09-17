from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

from src.project_paths import MODEL_PATH, get_figure_path

# Aesthetic styling
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["font.family"] = "sans-serif"


def load_artifact():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model artifact missing at '{MODEL_PATH}'. "
            "Please train the model first by running: python -m src.train 400"
        )

    artifact = joblib.load(MODEL_PATH)
    pipeline = artifact["pipeline"]
    scaler = pipeline.named_steps["scaler"]
    knn = pipeline.named_steps["knn"]

    X_train_scaled = knn._fit_X
    y_train = knn._y
    target_names = artifact["target_names"]
    feature_names = artifact["feature_names"]

    return (
        pipeline,
        scaler,
        knn,
        X_train_scaled,
        y_train,
        target_names,
        feature_names,
    )


def plot_feature_distributions(show: bool = True):
    """Generates boxplots of key discriminating diagnostic features."""
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["diagnosis"] = [data.target_names[i].upper() for i in data.target]

    selected_features = [
        "mean radius",
        "mean texture",
        "mean perimeter",
        "mean area",
        "mean concavity",
    ]

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    fig.suptitle(
        "Clinical Measurement Separations: Malignant vs Benign",
        fontsize=14,
        fontweight="bold",
        y=1.05,
    )

    for ax, feat in zip(axes, selected_features):
        sns.boxplot(
            data=df,
            x="diagnosis",
            y=feat,
            hue="diagnosis",
            legend=False,
            ax=ax,
            palette={"MALIGNANT": "#e63946", "BENIGN": "#2a9d8f"},
        )
        ax.set_title(feat.title(), fontsize=11, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("")

    plt.tight_layout()
    out_path = Path(get_figure_path("feature_distributions.png"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"[+] Saved feature boxplots to: {out_path.resolve()}")

    if show:
        plt.show()
    plt.close(fig)


def plot_pca_decision_space(show: bool = True):
    """Reduces 30D feature space to 2D PCA space and visualizes clusters & nearest neighbors."""
    (
        pipeline,
        scaler,
        knn,
        X_train_scaled,
        y_train,
        target_names,
        _,
    ) = load_artifact()

    pca = PCA(n_components=2, random_state=42)
    X_train_pca = pca.fit_transform(X_train_scaled)
    var_ratio = pca.explained_variance_ratio_ * 100

    raw_data = load_breast_cancer()
    _, X_test_raw, _, y_test = train_test_split(
        raw_data.data,
        raw_data.target,
        test_size=0.2,
        random_state=42,
        stratify=raw_data.target,
    )

    sample_indices = [0, 1, 2, 3, 4, 5]
    X_test_subset = X_test_raw[sample_indices]
    y_test_subset = y_test[sample_indices]

    X_test_scaled = scaler.transform(X_test_subset)
    X_test_pca = pca.transform(X_test_scaled)
    distances, neighbor_indices = knn.kneighbors(X_test_scaled)

    fig = plt.figure(figsize=(11, 7))
    colors = {"malignant": "#e63946", "benign": "#2a9d8f"}

    for class_idx, label in enumerate(target_names):
        mask = y_train == class_idx
        plt.scatter(
            X_train_pca[mask, 0],
            X_train_pca[mask, 1],
            c=colors[label],
            label=f"Training: {label.upper()}",
            alpha=0.6,
            s=70,
            edgecolors="none",
        )

    for i, test_pt_pca in enumerate(X_test_pca):
        plt.scatter(
            test_pt_pca[0],
            test_pt_pca[1],
            c="#ffb703",
            s=130,
            marker="*",
            edgecolors="black",
            linewidths=1.2,
            label="Unseen Test Point" if i == 0 else "",
        )
        for n_idx in neighbor_indices[i][:2]:
            nearest_pca = X_train_pca[n_idx]
            plt.plot(
                [test_pt_pca[0], nearest_pca[0]],
                [test_pt_pca[1], nearest_pca[1]],
                color="black",
                linestyle="--",
                linewidth=0.8,
                alpha=0.5,
            )

    plt.title(
        "2D PCA Projection: KNN Decision Space & Neighbor Vectors",
        fontsize=13,
        fontweight="bold",
    )
    plt.xlabel(f"Principal Component 1 ({var_ratio[0]:.1f}% Variance)")
    plt.ylabel(f"Principal Component 2 ({var_ratio[1]:.1f}% Variance)")
    plt.legend(frameon=True, loc="upper right")
    plt.tight_layout()

    out_path = Path(get_figure_path("knn_pca_decision_space.png"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"[+] Saved PCA cluster map to: {out_path.resolve()}")

    if show:
        plt.show()
    plt.close(fig)


def main():
    print("Generating biological & dimensional visualizations...")
    plot_feature_distributions(show=True)
    plot_pca_decision_space(show=True)
    print("\nVisualizations complete.")


if __name__ == "__main__":
    main()