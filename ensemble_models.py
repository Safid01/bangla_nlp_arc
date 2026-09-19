import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

def main():
    DATA_DIR = "./data"
    test_raw = pd.read_csv(os.path.join(DATA_DIR, "test.csv"), encoding="utf-8")
    sample_sub = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"), encoding="utf-8")
    train_raw = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), encoding="utf-8")
    val_raw = pd.read_csv(os.path.join(DATA_DIR, "validation.csv"), encoding="utf-8")

    target_cols = [c for c in sample_sub.columns if c.lower() != "id"]
    known_df = pd.concat([train_raw, val_raw], ignore_index=True)

    # 1. Load Prediction Matrices for all 3 architectures
    oof_buet = np.load("oof_banglabert_asl.npy")
    test_buet = np.load("test_banglabert_asl.npy")

    oof_sagor = np.load("oof_sagorbert_asl.npy")
    test_sagor = np.load("test_sagorbert_asl.npy")

    oof_qwen = np.load("oof_qwen.npy")
    test_qwen = np.load("test_qwen.npy")

    print("=" * 80)
    print("🚀 TRI-ARCHITECTURE ASL ENSEMBLE SUBMISSION GENERATOR")
    print("=" * 80)

    # 2. Optimal Tri-Architecture Blending
    oof_tri = 0.50 * oof_buet + 0.25 * oof_sagor + 0.25 * oof_qwen
    test_tri = 0.50 * test_buet + 0.25 * test_sagor + 0.25 * test_qwen

    labels_known = known_df[target_cols].values.astype(np.float32)

    # Calculate model predictions on test.csv using prevalence-matched thresholds
    test_preds = np.zeros_like(test_tri, dtype=int)
    calibrated_thresholds = []
    for i, col in enumerate(target_cols):
        prev = labels_known[:, i].mean()
        t = float(np.percentile(oof_tri[:, i], (1.0 - prev) * 100))
        calibrated_thresholds.append(t)
        test_preds[:, i] = (test_tri[:, i] >= t).astype(int)

    # Hierarchical rule: if any sub-label is 1, is_toxic is 1
    sub_cols_indices = [target_cols.index(c) for c in target_cols if c != "is_toxic"]
    any_sub_toxic = (test_preds[:, sub_cols_indices].sum(axis=1) > 0).astype(int)
    toxic_idx = target_cols.index("is_toxic")
    test_preds[:, toxic_idx] = np.maximum(test_preds[:, toxic_idx], any_sub_toxic)

    # Create submission dataframe with exact IDs from test.csv
    sub_df = pd.DataFrame(test_preds, columns=target_cols)
    sub_df.insert(0, "ID", test_raw["ID"])

    # Sanity & Integrity assertions
    assert (sub_df["ID"] == test_raw["ID"]).all(), "ID sequence mismatch with test.csv!"
    assert sub_df.shape == (len(test_raw), len(target_cols) + 1), "Shape mismatch!"
    assert sub_df.isna().sum().sum() == 0, "Null values found!"

    # Save to submission.csv
    sub_df.to_csv("submission.csv", index=False, encoding="utf-8")

    print(f"\n✅ Generated submission.csv with exact test.csv IDs ({len(sub_df)} rows)!")
    print(f"   First 5 IDs: {sub_df['ID'].head().tolist()}")
    print("\nSubmission Summary Statistics:")
    for col in target_cols:
        count = int(sub_df[col].sum())
        pct = (count / len(sub_df)) * 100
        print(f" - {col:20s}: {count:4d} positives ({pct:5.1f}%)")

if __name__ == "__main__":
    main()
