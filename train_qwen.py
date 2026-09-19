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
from peft import LoraConfig, TaskType, get_peft_model

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

def get_device_info():
    """Detects and returns optimal hardware device (Intel Arc XPU, CUDA, or CPU)."""
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        device_name = torch.xpu.get_device_name(0)
        print(f"🚀 Detected Intel Arc GPU: {device_name}")
        return "xpu", True
    elif torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        print(f"🚀 Detected NVIDIA GPU: {device_name}")
        return "cuda", True
    else:
        print("⚠️ No dedicated GPU detected. Falling back to CPU.")
        return "cpu", False

def clean_gpu_memory(device_type):
    """Safely frees VRAM across folds on Intel XPU or NVIDIA CUDA."""
    gc.collect()
    if device_type == "xpu" and hasattr(torch, "xpu"):
        torch.xpu.empty_cache()
    elif device_type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()

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

class WeightedTrainer(Trainer):
    def __init__(self, *args, pos_weight_tensor=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.pos_weight_tensor = pos_weight_tensor

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels").float()
        outputs = model(**inputs)
        logits = outputs.logits.float()
        weights = self.pos_weight_tensor.to(logits.device)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=weights)
        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss

def main():
    device_type, is_gpu = get_device_info()
    DATA_DIR = "./data"
    sample_sub = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"), encoding="utf-8")
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"), encoding="utf-8")
    aug_path = os.path.join(DATA_DIR, "train_counterfactual.csv")
    if os.path.exists(aug_path):
        all_train_df = pd.read_csv(aug_path, encoding="utf-8")
        full_train_df = all_train_df[all_train_df["is_augmented"] == 0].copy().reset_index(drop=True)
    else:
        train_raw = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), encoding="utf-8")
        val_raw = pd.read_csv(os.path.join(DATA_DIR, "validation.csv"), encoding="utf-8")
        full_train_df = pd.concat([train_raw, val_raw], ignore_index=True)

    target_cols = [c for c in sample_sub.columns if c.lower() != "id"]
    print(f"Target Labels ({len(target_cols)}): {target_cols}")
    print(f"Combined Training Data: {len(full_train_df)} rows | Test Data: {len(test_df)} rows")

    # 2. Bengali Normalization
    full_train_df["clean_text"] = full_train_df["text"].apply(clean_bangla)
    test_df["clean_text"] = test_df["text"].apply(clean_bangla)

    labels_matrix = full_train_df[target_cols].values.astype(np.float32)
    full_train_df["labels"] = list(labels_matrix)

    # 3. Stratified Multi-Label Folds
    N_SPLITS = 5
    full_train_df["strat_key"] = full_train_df[target_cols].astype(str).agg("".join, axis=1)
    key_counts = full_train_df["strat_key"].value_counts()
    rare_keys = key_counts[key_counts < N_SPLITS].index
    full_train_df.loc[full_train_df["strat_key"].isin(rare_keys), "strat_key"] = "rare"

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    full_train_df["fold"] = -1
    for fold, (_, val_idx) in enumerate(skf.split(full_train_df, full_train_df["strat_key"])):
        full_train_df.loc[val_idx, "fold"] = fold

    # 4. Tokenizer Setup (7B-Instruct)
    MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    def format_prompt(text):
        return f"<|im_start|>system\nYou are a Bengali NLP safety classifier. Analyze the text for toxicity, implicit toxicity, sarcasm, profanity, threats, identity attacks, and sexual content.<|im_end|>\n<|im_start|>user\nText: {text}<|im_end|>\n<|im_start|>assistant\n"

    def tokenize_func(batch):
        formatted = [format_prompt(t) for t in batch["clean_text"]]
        return tokenizer(formatted, truncation=True, max_length=192)

    test_ds = Dataset.from_pandas(test_df[["clean_text"]], preserve_index=False).map(tokenize_func, batched=True)
    test_ds.set_format(type="torch", columns=["input_ids", "attention_mask"])

    # 5. Cross-Validation Storage
    oof_predictions = np.zeros_like(labels_matrix, dtype=np.float32)
    test_predictions = np.zeros((len(test_df), len(target_cols)), dtype=np.float32)

    print("\n" + "=" * 65)
    print(f"🔥 STARTING {N_SPLITS}-FOLD QWEN2.5-7B-INSTRUCT TRAINING ON INTEL ARC")
    print("=" * 65)

    use_bf16 = True if device_type in ["cuda", "xpu"] else False

    for fold in range(N_SPLITS):
        print(f"\n>>> Running Fold {fold + 1} / {N_SPLITS} <<<")
        train_fold_df = full_train_df[full_train_df["fold"] != fold].reset_index(drop=True)
        val_fold_df = full_train_df[full_train_df["fold"] == fold].reset_index(drop=True)
        val_indices = full_train_df[full_train_df["fold"] == fold].index.values

        # Dynamic positive class weighting
        train_fold_labels = train_fold_df[target_cols].values.astype(np.float32)
        pos_counts = np.sum(train_fold_labels, axis=0)
        neg_counts = len(train_fold_labels) - pos_counts
        pos_weights = np.clip(neg_counts / (pos_counts + 1e-5), 1.0, 8.0)
        pos_weight_tensor = torch.tensor(pos_weights, dtype=torch.float32)

        train_ds = Dataset.from_pandas(train_fold_df[["clean_text", "labels"]], preserve_index=False).map(tokenize_func, batched=True)
        val_ds = Dataset.from_pandas(val_fold_df[["clean_text", "labels"]], preserve_index=False).map(tokenize_func, batched=True)

        train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
        val_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

        # Load 7B Model in Native BF16
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=len(target_cols),
            problem_type="multi_label_classification",
            dtype=torch.bfloat16 if use_bf16 else torch.float32,
        )
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.use_cache = False
        model.gradient_checkpointing_enable()

        # LoRA Configuration for 7B
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none",
        )
        model = get_peft_model(model, peft_config)

        # Keep trainable LoRA adapter weights in float32 for training stability
        for param in model.parameters():
            if param.requires_grad:
                param.data = param.data.float()

        training_args = TrainingArguments(
            output_dir=f"./qwen7b_intel_fold_{fold}",
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=1.5e-4,
            per_device_train_batch_size=2,         # Tuned for 16GB VRAM on 7B
            gradient_accumulation_steps=8,         # Effective Batch Size = 16
            per_device_eval_batch_size=4,
            num_train_epochs=3,
            warmup_steps=30,
            weight_decay=0.01,
            load_best_model_at_end=True,
            metric_for_best_model="mean_macro_f1",
            greater_is_better=True,
            logging_steps=20,
            report_to="none",
            bf16=use_bf16,                         # Native BF16 bypasses FP64 GradScaler crash
            fp16=False,                            # Explicitly disabled for Intel Arc stability
            gradient_checkpointing=True,
            optim="adamw_torch",
            dataloader_num_workers=0,              # Safe default on Windows
        )

        trainer = WeightedTrainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            data_collator=DataCollatorWithPadding(tokenizer=tokenizer, padding=True, return_tensors="pt"),
            compute_metrics=compute_metrics,
            pos_weight_tensor=pos_weight_tensor,
        )

        trainer.train()

        # Out-of-fold prediction
        val_raw = trainer.predict(val_ds)
        val_probs = expit(val_raw.predictions[0] if isinstance(val_raw.predictions, tuple) else val_raw.predictions)
        oof_predictions[val_indices] = val_probs

        # Test prediction accumulation
        test_raw = trainer.predict(test_ds)
        test_probs = expit(test_raw.predictions[0] if isinstance(test_raw.predictions, tuple) else test_raw.predictions)
        test_predictions += (test_probs / N_SPLITS)

        # Free GPU VRAM completely before the next fold
        del model, trainer, train_ds, val_ds
        clean_gpu_memory(device_type)

    np.save("oof_qwen.npy", oof_predictions)
    np.save("test_qwen.npy", test_predictions)
    print(f"\n✅ Saved oof_qwen.npy {oof_predictions.shape} and test_qwen.npy {test_predictions.shape} successfully.")

    # 6. Calibrate Thresholds Across All OOF Samples
    print("\n" + "=" * 65)
    print("⚙️ CALIBRATING THRESHOLDS ON 3,792 OOF SAMPLES")
    print("=" * 65)

    best_thresholds = []
    oof_f1_scores = []

    for i, col in enumerate(target_cols):
        best_t, best_f1 = 0.5, -1.0
        for t in np.arange(0.10, 0.75, 0.01):
            preds_col = (oof_predictions[:, i] >= t).astype(int)
            score = f1_score(labels_matrix[:, i], preds_col, average="macro", zero_division=0)
            if score > best_f1:
                best_f1 = score
                best_t = float(t)

        # Minority-class recall protection
        if col in ["implicit_toxicity", "threatening", "sexual_content", "identity_attack"]:
            best_t = min(best_t, 0.35)

        best_thresholds.append(best_t)
        oof_f1_scores.append(best_f1)
        print(f"Col: {col:20s} | Optimal Threshold: {best_t:.2f} | OOF Macro F1: {best_f1:.4f}")

    print("-" * 65)
    print(f"🏆 Final Out-of-Fold Mean Macro F1: {np.mean(oof_f1_scores):.4f}")
    print("-" * 65)

    # 7. Generate Submission File
    final_preds = np.zeros_like(test_predictions, dtype=int)
    for i, threshold in enumerate(best_thresholds):
        final_preds[:, i] = (test_predictions[:, i] >= threshold).astype(int)

    submission_df = sample_sub.copy()
    submission_df[target_cols] = final_preds
    submission_df.to_csv("submission.csv", index=False, encoding="utf-8")

    print("\n✅ submission.csv generated successfully!")
    print("\nTest Prediction Category Proportions:")
    print(submission_df.iloc[:, 1:].mean())

if __name__ == "__main__":
    main()