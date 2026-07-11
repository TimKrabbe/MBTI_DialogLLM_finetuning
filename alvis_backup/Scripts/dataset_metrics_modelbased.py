from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from sentence_transformers import SentenceTransformer, util
import torch
import gc
import pandas as pd
import numpy as np


# read data
bt_df = pd.read_csv("/cephyr/users/timkra/Alvis/MA/data/bt_augmented_agg.csv", sep = "\t", quoting = 1)
#bt_df = bt_df.rename(columns = {"post_augmented": "post_augmented_old", "post_augmented_john": "post_augmented"})

syn_df = pd.read_csv("/cephyr/users/timkra/Alvis/MA/data/syn_augmented_agg.csv" , sep = "\t", quoting = 1)
sw_df = pd.read_csv("/cephyr/users/timkra/Alvis/MA/data/sw_augmented_agg.csv", sep = "\t", quoting = 1)
del_df = pd.read_csv("/cephyr/users/timkra/Alvis/MA/data/del_augmented_agg.csv", sep = "\t", quoting = 1)

#syn_df = syn_df.drop(7788)

# calculate PPL
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
model.eval()

def PPL(df):
    texts = df["post_augmented"].tolist()
    nlls = []
    with torch.no_grad():
        for text in texts:
            encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
            loss = model(**encodings, labels=encodings["input_ids"]).loss
            nlls.append(loss.item())
    return float(np.exp(np.mean(nlls)))

bt_ppl = PPL(bt_df)
syn_ppl = PPL(syn_df)
sw_ppl = PPL(sw_df)
del_ppl = PPL(del_df)

print(f"PPL (Backtranslation): {bt_ppl}")
print(f"PPL (Synonym Swapping): {syn_ppl}")
print(f"PPL (Random Swapping): {sw_ppl}")
print(f"PPL (Random Deletion): {del_ppl}")

# syn_ppl check
texts = syn_df["post_augmented"].tolist()
nlls = []
with torch.no_grad():
    for i, text in enumerate(texts):
        encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
        loss = model(**encodings, labels=encodings["input_ids"]).loss
        val = loss.item()
        if np.isnan(val) or np.isinf(val):
            print(f"Problematic sentence {i}: {text[:100]}")
        nlls.append(val)

print(syn_df["post_augmented"].isna().sum())
print((syn_df["post_augmented"] == "").sum())


# empty gpu cache
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()


# calculate cosine similarity
model = SentenceTransformer("all-MiniLM-L6-v2")

def CosSim(df):
    original_embeddings = model.encode(df["post"].tolist(), batch_size=64, show_progress_bar=True)
    augmented_embeddings = model.encode(df["post_augmented"].tolist(), batch_size=64, show_progress_bar=True)

    cosine_scores = util.cos_sim(original_embeddings, augmented_embeddings).diagonal()
    return cosine_scores


bt_cs = CosSim(bt_df).mean()
syn_cs = CosSim(syn_df).mean()
sw_cs = CosSim(sw_df).mean()
del_cs = CosSim(del_df).mean()

print(f"Cosine Similarity (Backtranslation): {bt_cs}")
print(f"Cosine Similarity (Synonym Swapping): {syn_cs}")
print(f"Cosine Similarity (Random Swapping): {sw_cs}")
print(f"Cosine Similarity (Random Deletion): {del_cs}")


