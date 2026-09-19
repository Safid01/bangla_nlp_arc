import os
import sys
import gc
import re
import unicodedata
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.special import expit
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)
from asymmetric_loss import AsymmetricLossMultiLabel

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

def clean_bangla(text):
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[\u200B-\u200D\uFEFF]", "", text)
    return re.sub(r"\s+", " ", text).strip()

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    if isinstance(logits, tuple):
        logits = logits[0]
    probs = expit(logits)
    preds = (probs >= 0.5).astype(int)
    col_f1s = [
        f1_score(labels[:, i], preds[:, i], average="macro", zero_division=0)
        for i in range(labels.shape[1])
    ]
    return {"mean_macro_f1": float(np.mean(col_f1s))}

class ASLTrainer(Trainer):
    def __init__(self, *args, asl_loss_fn=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.asl_loss_fn = asl_loss_fn

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels").float()
        outputs = model(**inputs)
        logits = outputs.logits.float()
        loss = self.asl_loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss

def main():
    device_type = "xpu" if hasattr(torch, "xpu") and torch.xpu.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Training 5-Fold BanglaBERT with Asymmetric Focal Loss on: {device_type.upper()}")

    # 1. Load Augmented Dataset
    DATA_DIR = "./data"
    augmented_path = os.path.join(DATA_DIR, "train_augmented.csv")
    if not os.path.exists(augmented_path):
        raise FileNotFoundError(f"Missing {augmented_path}. Run generate_augmented_data.py first!")

    full_train_df = pd.read_csv(augmented_path, encoding="utf-8")
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"), encoding="utf-8")
    sample_sub = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"), encoding="utf-8")

    target_cols = [c for c in sample_sub.columns if c.lower() != "id"]
    print(f"Target Labels ({len(target_cols)}): {target_cols}")

    full_train_df["clean_text"] = full_train_df["clean_text"].fillna("").astype(str)
    test_df["clean_text"] = test_df["text"].apply(clean_bangla)

    # Separate original 3797 samples from augmented samples
    orig_mask = full_train_df["is_augmented"] == 0
    orig_train_df = full_train_df[orig_mask].copy().reset_index(drop=True)
    aug_train_df = full_train_df[~orig_mask].copy().reset_index(drop=True)

    print(f"Original samples: {len(orig_train_df)} | Augmented samples: {len(aug_train_df)} | Test: {len(test_df)}")

    labels_orig = orig_train_df[target_cols].values.astype(np.float32)
    orig_train_df["labels"] = list(labels_orig)

    labels_aug = aug_train_df[target_cols].values.astype(np.float32)
    aug_train_df["labels"] = list(labels_aug)

    # 2. Stratified 5-Folds on original samples
    N_SPLITS = 5
    orig_train_df["strat_key"] = orig_train_df[target_cols].astype(str).agg("".join, axis=1)
    key_counts = orig_train_df["strat_key"].value_counts()
    rare_keys = key_counts[key_counts < N_SPLITS].index
    orig_train_df.loc[orig_train_df["strat_key"].isin(rare_keys), "strat_key"] = "rare"

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    orig_train_df["fold"] = -1
    for fold, (_, val_idx) in enumerate(skf.split(orig_train_df, orig_train_df["strat_key"])):
        orig_train_df.loc[val_idx, "fold"] = fold

    # 3. Tokenizer
    BERT_NAME = "csebuetnlp/banglabert"
    print(f"Loading Tokenizer: {BERT_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(BERT_NAME)

    def tokenize_func(batch):
        return tokenizer(batch["clean_text"], truncation=True, max_length=160)

    test_ds = Dataset.from_pandas(test_df[["clean_text"]], preserve_index=False).map(
        tokenize_func, batched=True, remove_columns=["clean_text"]
    )
    test_ds.set_format(type="torch", columns=["input_ids", "attention_mask"])

    oof_bert = np.zeros_like(labels_orig, dtype=np.float32)
    test_bert = np.zeros((len(test_df), len(target_cols)), dtype=np.float32)

    # Asymmetric Loss configuration:
    # gamma_neg=2.5 suppresses easy negatives
    # gamma_pos=0.0 preserves positive gradients
    # clip=0.04 shifts negative probability margin
    asl_loss_fn = AsymmetricLossMultiLabel(gamma_neg=2.5, gamma_pos=0.0, clip=0.04)

    for fold in range(N_SPLITS):
        print("\n" + "=" * 60)
        print(f">>> Training BanglaBERT (ASL) Fold {fold + 1} / {N_SPLITS} <<<")
        print("=" * 60)

        val_fold_df = orig_train_df[orig_train_df["fold"] == fold].reset_index(drop=True)
        val_indices = orig_train_df[orig_train_df["fold"] == fold].index.values

        # Training set = (original training folds) + (all augmented counterfactual samples)
        orig_train_fold_df = orig_train_df[orig_train_df["fold"] != fold].reset_index(drop=True)
        train_fold_df = pd.concat([orig_train_fold_df, aug_train_df], ignore_index=True)

        print(f"Fold {fold+1} Training Samples: {len(train_fold_df)} | Validation Samples: {len(val_fold_df)}")

        train_ds = Dataset.from_pandas(train_fold_df[["clean_text", "labels"]], preserve_index=False).map(
            tokenize_func, batched=True, remove_columns=["clean_text"]
        )
        val_ds = Dataset.from_pandas(val_fold_df[["clean_text", "labels"]], preserve_index=False).map(
            tokenize_func, batched=True, remove_columns=["clean_text"]
        )

        train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
        val_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

        model = AutoModelForSequenceClassification.from_pretrained(
            BERT_NAME,
            num_labels=len(target_cols),
            problem_type="multi_label_classification",
        )

        output_fold_dir = f"./banglabert_asl_fold_{fold}"
        training_args = TrainingArguments(
            output_dir=output_fold_dir,
            eval_strategy="epoch",
            save_strategy="no",
            learning_rate=3e-5,
            per_device_train_batch_size=16,
            gradient_accumulation_steps=1,
            per_device_eval_batch_size=32,
            num_train_epochs=4,
            lr_scheduler_type="cosine",
            warmup_steps=80,
            weight_decay=0.01,
            logging_steps=25,
            report_to="none",
            bf16=(device_type == "xpu"),
            fp16=False,
            optim="adamw_torch",
            dataloader_num_workers=0,
        )

        trainer = ASLTrainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            data_collator=DataCollatorWithPadding(tokenizer=tokenizer, padding=True, return_tensors="pt"),
            compute_metrics=compute_metrics,
            asl_loss_fn=asl_loss_fn,
        )

        trainer.train()

        # In-memory Out-of-fold and Test Predictions
        val_raw = trainer.predict(val_ds)
        val_preds_logits = val_raw.predictions[0] if isinstance(val_raw.predictions, tuple) else val_raw.predictions
        oof_bert[val_indices] = expit(val_preds_logits)

        test_raw = trainer.predict(test_ds)
        test_preds_logits = test_raw.predictions[0] if isinstance(test_raw.predictions, tuple) else test_raw.predictions
        test_bert += (expit(test_preds_logits) / N_SPLITS)

        # Cleanup temporary output folder if created
        if os.path.exists(output_fold_dir):
            import shutil
            shutil.rmtree(output_fold_dir, ignore_errors=True)

        del model, trainer, train_ds, val_ds
        gc.collect()
        if device_type == "xpu":
            torch.xpu.empty_cache()
        elif device_type == "cuda":
            torch.cuda.empty_cache()

    # Save compact numpy prediction matrices
    np.save("oof_banglabert_asl.npy", oof_bert)
    np.save("test_banglabert_asl.npy", test_bert)
    print(f"\n✅ All 5 folds complete! Saved oof_banglabert_asl.npy {oof_bert.shape} and test_banglabert_asl.npy {test_bert.shape} successfully.")

    # Calculate and report BanglaBERT ASL standalone OOF score
    col_f1s = []
    print("\n" + "=" * 70)
    print("--- BanglaBERT (ASL) Standalone OOF Breakdown (at default t=0.5) ---")
    print("=" * 70)
    for i, col in enumerate(target_cols):
        preds = (oof_bert[:, i] >= 0.5).astype(int)
        actual = labels_orig[:, i]
        pos_preds = preds.sum()
        actual_pos = actual.sum()
        f1_p = f1_score(actual, preds, pos_label=1, zero_division=0)
        f1_n = f1_score(actual, preds, pos_label=0, zero_division=0)
        f1_m = f1_score(actual, preds, average="macro", zero_division=0)
        col_f1s.append(f1_m)
        print(f" - {col:20s}: Actual={actual_pos:4.0f}, Preds={pos_preds:4.0f} | F1_pos={f1_p:.4f}, F1_neg={f1_n:.4f} -> Macro={f1_m:.4f}")

    mean_f1 = np.mean(col_f1s)
    print("-" * 70)
    print(f"🏆 BanglaBERT (ASL) Mean Macro F1 (t=0.5): {mean_f1:.4f}")
    print("-" * 70)

if __name__ == "__main__":
    main()
