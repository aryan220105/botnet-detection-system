"""
Model Training Module for Botnet Detection System.

Trains two classifiers — Random Forest (supervised) and Isolation Forest
(anomaly-based) — on a labelled network traffic dataset and persists
them to disk with joblib.

The script can work with:
  * The bundled ``data/dataset_sample.csv`` (for quick demos)
  * The CICIDS-2017 or CTU-13 public datasets (full-scale training)

Usage
-----
    python models/train_model.py                       # use sample data
    python models/train_model.py --dataset path/to.csv # use custom CSV
"""

import os
import sys
import argparse
import json

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for servers
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    roc_auc_score,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from feature_engineering.feature_extractor import FEATURE_COLUMNS
from utils.logger import log_system

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")
DEFAULT_DATASET = os.path.join(DATA_DIR, "dataset_sample.csv")


# ──────────────────────────────────────────────
# Data loading & preprocessing
# ──────────────────────────────────────────────

def load_dataset(path: str) -> pd.DataFrame:
    """Load CSV dataset and perform basic sanity checks."""
    log_system(f"Loading dataset from {path}")
    df = pd.read_csv(path)
    log_system(f"Dataset shape: {df.shape}")
    log_system(f"Columns: {list(df.columns)}")
    log_system(f"Label distribution:\n{df['label'].value_counts().to_string()}")
    return df


def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Clean, select features, and return (X, y)."""
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = 0

    X = df[FEATURE_COLUMNS].copy()
    y = df["label"].copy()

    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(0, inplace=True)
    return X, y


# ──────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────

def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 100,
) -> RandomForestClassifier:
    """Train a Random Forest classifier for botnet detection."""
    log_system(f"Training RandomForest (n_estimators={n_estimators})...")
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=20,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    log_system("RandomForest training complete.")
    return clf


def train_isolation_forest(
    X_train: pd.DataFrame,
    contamination: float = 0.1,
) -> IsolationForest:
    """Train an Isolation Forest for unsupervised anomaly detection."""
    log_system(f"Training IsolationForest (contamination={contamination})...")
    iso = IsolationForest(
        n_estimators=150,
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
    )
    iso.fit(X_train)
    log_system("IsolationForest training complete.")
    return iso


# ──────────────────────────────────────────────
# Evaluation & visualisation
# ──────────────────────────────────────────────

def evaluate_model(
    clf,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = "RandomForest",
) -> dict:
    """Print and return evaluation metrics."""
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    report = classification_report(y_test, y_pred, zero_division=0)
    log_system(f"\n{'='*50}\n{model_name} Evaluation\n{'='*50}")
    log_system(f"Accuracy : {acc:.4f}")
    log_system(f"F1 Score : {f1:.4f}")
    log_system(f"\n{report}")

    print(f"\n{'='*50}")
    print(f"{model_name} Evaluation")
    print(f"{'='*50}")
    print(f"Accuracy : {acc:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"\n{report}")

    return {"accuracy": acc, "f1_score": f1}


def plot_confusion_matrix(
    clf,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = "RandomForest",
) -> None:
    """Save a confusion-matrix heatmap to the models/ directory."""
    y_pred = clf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Normal", "Botnet"],
                yticklabels=["Normal", "Botnet"])
    plt.title(f"{model_name} - Confusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    path = os.path.join(MODEL_DIR, f"{model_name.lower()}_confusion_matrix.png")
    plt.savefig(path, dpi=150)
    plt.close()
    log_system(f"Confusion matrix saved to {path}")


def plot_feature_importance(clf, model_name: str = "RandomForest") -> None:
    """Save a feature-importance bar chart."""
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    names = [FEATURE_COLUMNS[i] for i in indices]

    plt.figure(figsize=(8, 5))
    sns.barplot(x=importances[indices], y=names, palette="viridis")
    plt.title(f"{model_name} - Feature Importance")
    plt.xlabel("Importance")
    plt.tight_layout()
    path = os.path.join(MODEL_DIR, f"{model_name.lower()}_feature_importance.png")
    plt.savefig(path, dpi=150)
    plt.close()
    log_system(f"Feature importance chart saved to {path}")


# ──────────────────────────────────────────────
# Persistence
# ──────────────────────────────────────────────

def save_model(model, scaler, model_name: str = "random_forest") -> str:
    """Save trained model + scaler bundle to disk."""
    bundle = {"model": model, "scaler": scaler, "features": FEATURE_COLUMNS}
    path = os.path.join(MODEL_DIR, f"{model_name}_model.pkl")
    joblib.dump(bundle, path)
    log_system(f"Model saved to {path}")
    print(f"Model saved -> {path}")
    return path


# ──────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────

def main(dataset_path: str = DEFAULT_DATASET) -> None:
    print("=" * 60)
    print("  Botnet Detection System - Model Training Pipeline")
    print("=" * 60)

    df = load_dataset(dataset_path)
    X, y = preprocess(df)

    # Normalise features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=FEATURE_COLUMNS)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.25, random_state=42, stratify=y,
    )

    # ---- Random Forest (supervised) ----
    rf_clf = train_random_forest(X_train, y_train)
    evaluate_model(rf_clf, X_test, y_test, "RandomForest")
    plot_confusion_matrix(rf_clf, X_test, y_test, "RandomForest")
    plot_feature_importance(rf_clf, "RandomForest")
    save_model(rf_clf, scaler, "random_forest")

    # ---- Isolation Forest (anomaly) ----
    iso_clf = train_isolation_forest(X_train, contamination=0.15)

    # IsolationForest returns -1/1; map to 1/0 for comparison with labels
    iso_pred = iso_clf.predict(X_test)
    iso_labels = pd.Series(np.where(iso_pred == -1, 1, 0), index=y_test.index)
    iso_acc = accuracy_score(y_test, iso_labels)
    print(f"\nIsolationForest Accuracy: {iso_acc:.4f}")
    log_system(f"IsolationForest Accuracy: {iso_acc:.4f}")

    save_model(iso_clf, scaler, "isolation_forest")

    # Save training metrics for the dashboard
    metrics = {
        "random_forest_accuracy": round(float(accuracy_score(y_test, rf_clf.predict(X_test))), 4),
        "isolation_forest_accuracy": round(float(iso_acc), 4),
        "training_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "features_used": FEATURE_COLUMNS,
    }
    metrics_path = os.path.join(MODEL_DIR, "training_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Training metrics saved -> {metrics_path}")

    print("\n[OK] Training pipeline complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train botnet detection models")
    parser.add_argument(
        "--dataset", type=str, default=DEFAULT_DATASET,
        help="Path to labelled CSV dataset",
    )
    args = parser.parse_args()
    main(args.dataset)
