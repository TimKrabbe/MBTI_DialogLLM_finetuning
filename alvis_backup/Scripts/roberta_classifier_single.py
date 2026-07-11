import torch
from datasets import load_dataset, Dataset, Features, ClassLabel, Sequence, Value
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, AutoConfig, EarlyStoppingCallback
import evaluate
import numpy as np
from huggingface_hub import login
from sklearn.metrics import f1_score as sklearn_f1
import os


login(token=os.environ["HF_TOKEN"])

model_id = "roberta-large"
df_id = "DrinkIcedT/MBTI_agg_balanced"
repo_id = "DrinkIcedT/roberta-large_MBTI"
rseed=42

df = load_dataset(df_id)
df = df.rename_column("labels", "complete_label")


tokenizer = AutoTokenizer.from_pretrained(model_id)
def tokenize(batch):
     return tokenizer(batch["post"], padding="max_length", truncation=True, max_length=512)


# ignore
#################################################################################
# token_lengths = [len(tokenizer.encode(text)) for text in df["train"]["post"]]
# print(f"Max: {max(token_lengths)}")
# print(f"Mean: {np.mean(token_lengths):.0f}")
# print(f"Median: {np.median(token_lengths):.0f}")
# print(f"% über 256: {(np.array(token_lengths) > 256).mean()*100:.1f}%")
# print(f"% über 512: {(np.array(token_lengths) > 512).mean()*100:.1f}%")
# print(f"% über 600: {(np.array(token_lengths) > 600).mean()*100:.1f}%")

## alt
#def compute_metrics_macro(eval_pred):
#    logits, labels = eval_pred
#    probs = 1 / (1 + np.exp(-logits.squeeze()))
#    predictions = (probs > 0.5).astype(int)
#    labels = labels.astype(int).squeeze()
#    
#    f1 = sklearn_f1(labels, predictions, average="macro")
#    acc = (predictions == labels).mean()
#    
#    return {"f1": float(f1), "acc": float(acc)}

# auch alt, aber neuer
#def compute_metrics_macro(eval_pred):
#    logits, labels = eval_pred
#    probs = 1 / (1 + np.exp(-logits.squeeze()))
#    predictions = (probs > 0.5).astype(int)
#    labels = labels.astype(int).squeeze()
#    
#    
#    f1 = sklearn_f1(labels, predictions, average="macro")
#    acc = (predictions == labels).mean()
#    return {"f1": float(f1), "acc": float(acc)}

# hyperparameter tuning, but not detailed enough
#def compute_metrics_macro(eval_pred):
#    logits, labels = eval_pred
#    # Wahrscheinlichkeiten berechnen
#    probs = 1 / (1 + np.exp(-logits.squeeze()))
#    labels = labels.astype(int).squeeze()
#    
#    # 1. Standard-Metrik bei 0.5
#    predictions_05 = (probs > 0.5).astype(int)
#    f1_05 = sklearn_f1(labels, predictions_05, average="macro")
#    acc_05 = (predictions_05 == labels).mean()
#    
#    # 2. Analyse: Verschiedene Thresholds testen (wird in die Konsole gedruckt)
#    print("\n--- Threshold Analyse ---")
#    best_f1_val = 0
#    best_t = 0.5
#    
#    for t in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
#        current_preds = (probs > t).astype(int)
#        current_f1 = sklearn_f1(labels, current_preds, average="macro")
#        print(f"Threshold {t:.1f} | F1-Macro: {current_f1:.4f}")
#        
#        if current_f1 > best_f1_val:
#            best_f1_val = current_f1
#            best_t = t
#    print(f"-> Bester F1 möglich: {best_f1_val:.4f} bei Threshold {best_t}")
#    print(f"-> Preds Mean (bei 0.5): {predictions_05.mean():.4f} | Labels Mean: {labels.mean():.4f}")
#    print("------------------------\n")
#
#    # Wir geben den Standard-F1 zurück, damit das Training vergleichbar bleibt,
#    # fügen aber den "besten möglichen" F1 als Info hinzu.
#    return {
#        "f1": float(f1_05), 
#        "f1_optimized": float(best_f1_val),
#        "best_threshold": float(best_t),
#        "acc": float(acc_05)
#    }
##################################################################################################
#end of ignoring ;)


def compute_metrics_macro(eval_pred):
    logits, labels = eval_pred
    # probs (for num_labels=1)
    probs = 1 / (1 + np.exp(-logits.squeeze()))
    labels = labels.astype(int).squeeze()
    
    # threshold tuning
    best_f1 = 0
    best_t = 0.5
    
    # np.linspace --> 0.50, 0.51, 0.53 ... 0. 85
    for t in np.linspace(0.5, 0.85, 36):
        preds = (probs > t).astype(int)
        f1 = sklearn_f1(labels, preds, average="macro")
        if f1 > best_f1:
            best_f1 = f1
            #print(f"Pred mean: {preds.mean():.4f}")  # 1.0 = immer positiv
            #print(f"Label mean: {labels.mean():.4f}")
            #print(f"Unique preds: {np.unique(preds)}")  # nur [1] = Majority Class
            best_t = t
            
    return {
        "f1": float(best_f1),           # best f1 I got
        "threshold": float(best_t),     # optimal t
        "f1_at_05": float(sklearn_f1(labels, (probs > 0.5).astype(int), average="macro"))
    }



DIMENSIONS = ["P"]

for dim in DIMENSIONS:
    print(f"\n=== Training classifier for dimension: {dim} ===")
    
    # reduce labels
    def make_single_label(example):
        example["labels"] = float(example[dim])
        return example
    
    train_dim = df["train"].map(make_single_label)
    val_dim   = df["validation"].map(make_single_label)
    test_dim  = df["test"].map(make_single_label)
    
    # remove cols
    cols_to_remove = ["I", "N", "F", "P", "complete_label"]
    train_dim = train_dim.remove_columns(cols_to_remove)
    val_dim   = val_dim.remove_columns(cols_to_remove)
    test_dim  = test_dim.remove_columns(cols_to_remove)
    
    # tokenization
    train_dim = train_dim.map(tokenize, batched=True, batch_size=1000)
    val_dim   = val_dim.map(tokenize,   batched=True, batch_size=1000)
    test_dim  = test_dim.map(tokenize,  batched=True, batch_size=1000)
    
    train_dim.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    val_dim.set_format("torch",   columns=["input_ids", "attention_mask", "labels"])
    test_dim.set_format("torch",  columns=["input_ids", "attention_mask", "labels"])

    
    # Model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        num_labels=1,
        problem_type="regression"  # BCE erwartet num_labels=1
    )
    
    # Training Arguments
    repo_dim = f"DrinkIcedT/roberta-large_MBTI_{dim}"
    training_args = TrainingArguments(
        output_dir=repo_dim,
        num_train_epochs=5,
        eval_strategy="steps",
        eval_steps=200,
        logging_steps=50,
        learning_rate=1e-5,
        weight_decay=0.01,
        warmup_steps=400,
	max_grad_norm=1.0,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        bf16=True,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        hub_strategy="end",
        report_to="tensorboard",
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dim,
        eval_dataset=val_dim,
        compute_metrics=compute_metrics_macro,
#	callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    trainer.train()
    
    # Test
    test_results = trainer.predict(test_dim)
    print(f"{dim} test results:", test_results.metrics)
    
    trainer.push_to_hub()


