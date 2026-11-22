"""Evaluate PyOD's unsupervised AutoEncoder on the specified Kaggle dataset."""

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pyod.models.auto_encoder import AutoEncoder
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler


DATASET_URL = "https://www.kaggle.com/datasets/whenamancodes/fraud-detection"
DATASET_SHA256 = "76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89"
FEATURES = ["Time", *(f"V{i}" for i in range(1, 29)), "Amount"]
SEED = 42


def load_data(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load only the verified version 1 Kaggle CSV, not a similarly named dataset."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {path}. Download {DATASET_URL} and extract creditcard.csv "
            "to data/creditcard.csv."
        )

    # Check the file contents, since the filename alone does not establish provenance.
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != DATASET_SHA256:
        raise ValueError(
            f"Unexpected CSV checksum for {path}. Expected the version 1 archive "
            f"from {DATASET_URL} (SHA-256: {DATASET_SHA256})."
        )

    data = pd.read_csv(path)
    if set(data.columns) != set(FEATURES + ["Class"]) or len(data) != 284_807:
        raise ValueError("Unexpected columns or transaction count in Kaggle creditcard.csv.")
    if data.isna().any().any() or not np.isfinite(data[FEATURES].to_numpy()).all():
        raise ValueError("Missing or non-finite transaction values cannot be used for training.")
    if not data["Class"].isin((0, 1)).all() or int(data["Class"].sum()) != 492:
        raise ValueError("Unexpected fraud labels or count in Kaggle creditcard.csv.")
    return data[FEATURES], data["Class"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(__file__).parent / "data" / "creditcard.csv",
        help="Path to creditcard.csv from the specified Kaggle dataset (version 1).",
    )
    args = parser.parse_args()

    features, labels = load_data(args.data)
    # Stratification reserves comparable fraud proportions for evaluation;
    # the labels are never supplied to the unsupervised model.
    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=SEED, stratify=labels
    )
    # Fit scaling on training rows only; test transactions must stay unseen.
    scaler = RobustScaler()
    train_scaled = scaler.fit_transform(x_train).astype(np.float32)
    test_scaled = scaler.transform(x_test).astype(np.float32)

    torch.manual_seed(SEED)
    # A fixed contamination rate determines the training-score cutoff without test labels.
    model = AutoEncoder(
        contamination=0.002,
        preprocessing=False,
        epoch_num=5,
        batch_size=512,
        hidden_neuron_list=[32, 16],
        device="cpu",
        random_state=SEED,
        verbose=0,
    )
    model.fit(train_scaled)
    # Continuous reconstruction scores assess ranking; predictions use the fitted cutoff.
    scores = model.decision_function(test_scaled)
    predictions = model.predict(test_scaled)
    tn, fp, fn, tp = confusion_matrix(y_test, predictions, labels=[0, 1]).ravel()

    print(f"Dataset: {DATASET_URL} (version 1)")
    print(f"CSV SHA-256: {DATASET_SHA256}")
    print(f"Transactions: {len(features):,}; fraud: {int(labels.sum()):,}")
    print(f"Training: {len(x_train):,}; test: {len(x_test):,}; test fraud: {int(y_test.sum()):,}")
    print("Model: PyOD AutoEncoder; 5 epochs; contamination: 0.002; seed: 42")
    print(f"Training score threshold: {model.threshold_:.6f}")
    print(f"Average precision (AP): {average_precision_score(y_test, scores):.6f}")
    print(f"ROC-AUC: {roc_auc_score(y_test, scores):.6f}")
    print(f"Fraud precision: {precision_score(y_test, predictions, zero_division=0):.6f}")
    print(f"Fraud recall: {recall_score(y_test, predictions, zero_division=0):.6f}")
    print(f"Confusion matrix: TN={tn:,} FP={fp:,} FN={fn:,} TP={tp:,}")
    print(f"Random precision baseline (test fraud rate): {y_test.mean():.6f}")


if __name__ == "__main__":
    main()
