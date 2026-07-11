import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
from tqdm import tqdm
import os
from huggingface_hub import login

# login
login(token=os.environ["HF_TOKEN"])

############ config
DIMENSIONS = ["I", "N", "F", "P"]
PAIRS = {"I": ("I","E"), "N": ("N","S"), "F": ("F","T"), "P": ("P","J")}
THRESHOLDS = {"I": 0.71, "N": 0.61, "F": 0.62, "P": 0.63}  # deine optimalen Thresholds eintragen
BATCH_SIZE = 32
HF_REPO = "DrinkIcedT/roberta-large_MBTI"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

############ load models, one roberta model per dimension (possible with lots of vram)
print("Loading models...")
tokenizers = {}
models = {}
for dim in DIMENSIONS:
    repo = f"{HF_REPO}_{dim}"
    tokenizers[dim] = AutoTokenizer.from_pretrained("roberta-large")
    models[dim] = AutoModelForSequenceClassification.from_pretrained(repo).to(DEVICE)
    models[dim].eval()
print("All models loaded.")

# data
df_base = pd.read_csv("/cephyr/users/timkra/Alvis/MA/data/base_test_modifiedprompt_10.csv")
df_tuned = pd.read_csv("/cephyr/users/timkra/Alvis/MA/data/tuned_test_modifiedprompt_10.csv")

texts_base = df_base["text"].tolist()
texts_tuned = df_tuned["text"].tolist()

records_base = (
    [{"text": t, "idx": i} for i, t in enumerate(texts_base)]
)
texts_base = [r["text"] for r in records_base]

records_tuned = (
    [{"text": t, "idx": i} for i, t in enumerate(texts_tuned)]
)
texts_tuned = [r["text"] for r in records_tuned]

######### batch inference
def predict_batch(texts, dim):
    all_probs = []
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc=f"Predicting {dim}"):
        batch = texts[i:i+BATCH_SIZE]
        inputs = tokenizers[dim](
            batch,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding="max_length"
        ).to(DEVICE)
        
        with torch.no_grad():
            logits = models[dim](**inputs).logits
            # sigmoid
            # .view(-1) flattens tensor, no nested lists
            probs = torch.sigmoid(logits).view(-1)
            
            batch_probs = probs.cpu().tolist()
            
        all_probs.extend(batch_probs)
    return all_probs

###### building mbti labels
def build_label(probs_per_dim, idx):
    label = ""
    for dim, (pos, neg) in PAIRS.items():
        label += pos if probs_per_dim[dim][idx] > THRESHOLDS[dim] else neg
    return label

def predict_labels(records, texts):
##### labelling 
    print("Running inference...")
    probs_per_dim = {}
    for dim in DIMENSIONS:
        probs_per_dim[dim] = predict_batch(texts, dim)

    labels = [build_label(probs_per_dim, i) for i in range(len(texts))]

    ###### build dataset
    result_df = pd.DataFrame({
        "idx": [r["idx"] for r in records],
        "text": texts,
        "label": labels,
        **{f"prob_{dim}": probs_per_dim[dim] for dim in DIMENSIONS}
    })

    return result_df

df_base = predict_labels(records_base, texts_base)
df_tuned = predict_labels(records_tuned, texts_tuned)


df_base.to_csv("/cephyr/users/timkra/Alvis/MA/data/base_test_labeled_10.csv", index=False)
df_tuned.to_csv("/cephyr/users/timkra/Alvis/MA/data/tuned_test_labeled_10.csv", index=False)
