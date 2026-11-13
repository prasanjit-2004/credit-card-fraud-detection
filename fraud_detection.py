"""
Credit Card Fraud Detection
---------------------------
Trains and compares Logistic Regression and Random Forest models on the
highly imbalanced Kaggle "Credit Card Fraud Detection" dataset.

Dataset: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
Place the downloaded file at: data/creditcard.csv

Usage:
    python fraud_detection.py
    python fraud_detection.py --data path/to/creditcard.csv
"""

import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    average_precision_score,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

RANDOM_STATE = 42
OUTPUT_DIR = Path("outputs")


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Loaded {df.shape[0]:,} rows and {df.shape[1]} columns")
    return df


def explore(df: pd.DataFrame) -> None:
    print("\n--- Data overview ---")
    print("Missing values:", int(df.isnull().sum().sum()))
    print("Duplicate rows:", int(df.duplicated().sum()))
    counts = df["Class"].value_counts()
    print("Class counts:\n", counts)
    print(f"Fraud percentage: {counts.get(1, 0) / len(df) * 100:.4f}%")

    OUTPUT_DIR.mkdir(exist_ok=True)
    ax = counts.plot(kind="bar", color=["#34d399", "#f87171"])
    ax.set_title("Class distribution (0 = legit, 1 = fraud)")
    ax.set_xlabel("Class")
    ax.set_ylabel("Transactions")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "class_distribution.png", dpi=150)
    plt.close()


def preprocess(df: pd.DataFrame):
    df = df.drop_duplicates().copy()

    # Amount and Time are not scaled like the PCA features V1-V28.
    # RobustScaler is less sensitive to outliers than StandardScaler.
    scaler = RobustScaler()
    df["Amount"] = scaler.fit_transform(df[["Amount"]])
    df["Time"] = scaler.fit_transform(df[["Time"]])

    X = df.drop(columns=["Class"])
    y = df["Class"]

    # Stratify so train and test keep the same (tiny) fraud ratio.
    return train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )


def build_models() -> dict:
    # class_weight="balanced" handles the imbalance without resampling.
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
    }


def evaluate(name, model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)  # best metric for imbalance

    print(f"\n=== {name} ===")
    print(classification_report(y_test, y_pred, digits=4))
    print(f"ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f}")

    slug = name.lower().replace(" ", "_")
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap="Blues")
    plt.title(f"Confusion matrix - {name}")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"confusion_matrix_{slug}.png", dpi=150)
    plt.close()

    return {"name": name, "roc_auc": roc_auc, "pr_auc": pr_auc, "y_prob": y_prob}


def plot_curves(results, y_test) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for r in results:
        RocCurveDisplay.from_predictions(
            y_test, r["y_prob"], name=r["name"], ax=axes[0]
        )
        PrecisionRecallDisplay.from_predictions(
            y_test, r["y_prob"], name=r["name"], ax=axes[1]
        )
    axes[0].set_title("ROC curve")
    axes[1].set_title("Precision-Recall curve")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "model_curves.png", dpi=150)
    plt.close()


def plot_feature_importance(model, columns) -> None:
    importances = pd.Series(model.feature_importances_, index=columns)
    top = importances.sort_values(ascending=False).head(15)[::-1]
    top.plot(kind="barh", color="#3b82f6", figsize=(7, 6))
    plt.title("Top 15 features - Random Forest")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "feature_importance.png", dpi=150)
    plt.close()


def main(data_path: str) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    df = load_data(data_path)
    explore(df)
    X_train, X_test, y_train, y_test = preprocess(df)

    results, trained = [], {}
    for name, model in build_models().items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        trained[name] = model
        results.append(evaluate(name, model, X_test, y_test))

    plot_curves(results, y_test)
    plot_feature_importance(trained["Random Forest"], X_train.columns)

    # Save the model with the best PR-AUC.
    best = max(results, key=lambda r: r["pr_auc"])
    joblib.dump(trained[best["name"]], OUTPUT_DIR / "best_model.joblib")
    print(f"\nBest model: {best['name']} (PR-AUC {best['pr_auc']:.4f})")
    print(f"Saved to {OUTPUT_DIR / 'best_model.joblib'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Credit card fraud detection")
    parser.add_argument("--data", default="data/creditcard.csv", help="CSV path")
    args = parser.parse_args()
    main(args.data)
