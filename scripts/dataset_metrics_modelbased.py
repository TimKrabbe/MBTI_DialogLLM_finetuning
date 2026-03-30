import evaluate
from sentence_transformers import SentenceTransformer, util
import torch
import gc
import pandas as pd

# read data
bt_df = pd.read_csv("~/MA/data/bt_augmented_john.csv")
bt_df = bt_df.rename(columns = {"post_augmented": "post_augmented_old", "post_augmented_john": "post_augmented"})

syn_df = pd.read_csv("~/MA/data/syn_augmented_ver4.csv")
sw_df = pd.read_csv("~/MA/data/sw_augmented_ver3.csv")
del_df = pd.read_csv("~/MA/data/del_augmented_ver3.csv")

# calculate PPL
perplexity = evaluate.load("perplexity", module_type="metric")

def PPL(df):
    results = perplexity.compute(
        predictions=df["post_augmented"].tolist(),
        model_id="gpt2"
    )
    return(results["mean_perplexity"])

bt_ppl = PPL(bt_df)
syn_ppl = PPL(syn_df)
sw_ppl = PPL(sw_df)
del_ppl = PPL(del_df)

print(f"PPL (Backtranslation): {bt_ppl}")
print(f"PPL (Synonym Swapping): {syn_ppl}")
print(f"PPL (Random Swapping): {sw_ppl}")
print(f"PPL (Random Deletion): {del_ppl}")



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


