####### decided to do all of it in one script instead of multiple files (notebook, script)

from transformers import MarianMTModel, MarianTokenizer, BertTokenizer, AutoTokenizer
from datasets import load_dataset
import nlpaug.augmenter.word as naw
import pandas as pd
import torch
import nltk
import gc
from tqdm import tqdm
import re
from huggingface_hub import login
import os

tqdm.pandas()

rseed = 42
login(token=os.environ.get("HF_TOKEN"))

def patch_tokenizer():
    if not hasattr(BertTokenizer, '_convert_token_to_id'):
        BertTokenizer._convert_token_to_id = lambda self, token: self.convert_tokens_to_ids(token)
    print("Tokenizer-Patch angewendet!")

patch_tokenizer()

PLACEHOLDER = "John"

def protect_mbti_mask(text):
    text = text.replace("<mbti>", PLACEHOLDER)
    return text

def restore_mbti_mask(text):
    text = re.sub(rf"{PLACEHOLDER}", "<mbti>", text, flags=re.IGNORECASE)
    return text

def backtranslate_safe(texts, device="cuda"):
    # load models
    en_de_tok = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-de")
    en_de_mod = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-de").to(device)
    de_en_tok = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-de-en")
    de_en_mod = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-de-en").to(device)

    gen_kwargs = dict(
        top_p=0.95,
        temperature=0.5,
        repetition_penalty=1.1,
        no_repeat_ngram_size=3,
        max_length=512,
        do_sample=True,
    )

    def translate(model, tokenizer, batch):
        if not batch: return []
        inputs = tokenizer(batch, return_tensors="pt", padding=True,
                          truncation=True, max_length=512).to(device)
        with torch.no_grad():
            out = model.generate(**inputs, **gen_kwargs)
        return tokenizer.batch_decode(out, skip_special_tokens=True)

    #### split texts
    all_sentences_to_translate = []
    group_sizes = []

    for text in texts:
        protected_text = protect_mbti_mask(text)
        parts = [p.strip() for p in protected_text.split("</s>") if p.strip()]
        all_sentences_to_translate.extend(parts)
        group_sizes.append(len(parts))

    #### batch translation
    translated_results = []
    batch_size = 16
    
    print(f"Übersetze {len(all_sentences_to_translate)} Einzel-Posts...")
    for i in tqdm(range(0, len(all_sentences_to_translate), batch_size)):
        batch = all_sentences_to_translate[i:i+batch_size]
        
        # backtranslation
        de = translate(en_de_mod, en_de_tok, batch)
        en_back = translate(de_en_mod, de_en_tok, de)
        
        translated_results.extend(en_back)

    # empty cache
    del en_de_mod, de_en_mod
    gc.collect()
    torch.cuda.empty_cache()

    #### build back together grouped by author
    final_texts = []
    current_idx = 0
    for size in group_sizes:
        group = translated_results[current_idx : current_idx + size]
        combined = " </s> ".join(group)
        # placeholder
        final_texts.append(restore_mbti_mask(combined))
        current_idx += size

    return final_texts

def random_swap(text, aug):
    augmented_text = aug.augment(text)
    return augmented_text[0] if isinstance(augmented_text, list) else augmented_text

def random_del(text, aug):
    augmented_text = aug.augment(text)
    return augmented_text[0] if isinstance(augmented_text, list) else augmented_text

def label_samples(df, label, n_goal, rseed):
    # count orginals
    obs_count = (df["labels"] == label).sum()
    
    if obs_count < n_goal:
        n_diff = n_goal - obs_count
        
        # only text longer than 5 chars
        long_posts = df[df["post"].str.split().str.len() >= 5]
        df_subset = long_posts[long_posts["labels"] == label]

        # fallback
        if df_subset.empty:
            df_subset = df[df["labels"] == label]

        n_syn = round(n_diff * 0.2)
        n_bt = round(n_diff * 0.7)
        n_sw = round(n_diff * 0.05)
        n_del = n_diff - (n_syn + n_bt + n_sw)

        s1 = df_subset.sample(n=n_syn, replace=True, random_state=rseed)
        s2 = df_subset.sample(n=n_bt, replace=True, random_state=rseed)
        s3 = df_subset.sample(n=n_sw, replace=True, random_state=rseed)
        s4 = df_subset.sample(n=n_del, replace=True, random_state=rseed)

        return s1, s2, s3, s4
    else:
        empty = df.iloc[:0]
        return empty, empty, empty, empty

df_hfdict = load_dataset("DrinkIcedT/mbti_unbalanced")
df = df_hfdict["train"].to_pandas()


syn_sample = pd.DataFrame(columns=df.columns)
bt_sample = pd.DataFrame(columns=df.columns)
sw_sample = pd.DataFrame(columns=df.columns)
del_sample = pd.DataFrame(columns=df.columns)

for label in df["labels"].unique():
    s1, s2, s3, s4 = label_samples(df, label, n_goal=2000, rseed=42)
    syn_sample = pd.concat([syn_sample, s1])
    bt_sample = pd.concat([bt_sample, s2])
    sw_sample = pd.concat([sw_sample, s3])
    del_sample = pd.concat([del_sample, s4])

# Dropna
syn_sample = syn_sample.dropna(subset=["post"])
bt_sample = bt_sample.dropna(subset=["post"])
sw_sample = sw_sample.dropna(subset=["post"])
del_sample = del_sample.dropna(subset=["post"])

def eda(syn_sample, sw_sample, del_sample):
    def augment_grouped_text(texts, augment_func):
        all_parts = []
        group_sizes = []
        
        # split
        for text in texts:
            protected = protect_mbti_mask(text)
            parts = [p.strip() for p in protected.split("</s>") if p.strip()]
            all_parts.extend(parts)
            group_sizes.append(len(parts))
        
        # augmentation
        augmented_parts = augment_func(all_parts)
        
        # building back
        reconstructed = []
        idx = 0
        for size in group_sizes:
            group = augmented_parts[idx : idx + size]
            combined = " </s> ".join(group)
            reconstructed.append(restore_mbti_mask(combined))
            idx += size
        return reconstructed

    # synonmy swapping
    aug_syn_bert = naw.ContextualWordEmbsAug(
        model_path='bert-base-uncased',
        action="substitute",
        device="cuda",
        stopwords=[PLACEHOLDER],
        aug_p=0.1,
        top_k=5,
    )
    
    print("Starte BERT Substitution...")
    syn_sample["post_augmented"] = augment_grouped_text(
        syn_sample["post"].tolist(), 
        lambda x: aug_syn_bert.augment(x)
    )
    
    del aug_syn_bert
    gc.collect()
    torch.cuda.empty_cache()

    # random swapping
    aug_rand_swap = naw.RandomWordAug(action="swap", stopwords=[PLACEHOLDER])
    
    print("Starte Random Swap...")
    sw_sample["post_augmented"] = augment_grouped_text(
        sw_sample["post"].tolist(),
        lambda x: [aug_rand_swap.augment(p)[0] for p in x]
    )
    
    # Random Deletion
    aug_rand_del = naw.RandomWordAug(action="delete", stopwords=[PLACEHOLDER])
    
    print("Starte Random Deletion...")
    del_sample["post_augmented"] = augment_grouped_text(
        del_sample["post"].tolist(),
        lambda x: [aug_rand_del.augment(p)[0] for p in x]
    )

    return syn_sample, sw_sample, del_sample

syn_df, sw_df, del_df = eda(syn_sample, sw_sample, del_sample)

texts = bt_sample["post"].tolist()
bt_sample["post_augmented"] = backtranslate_safe(texts)

# safe data
bt_sample.to_csv("~/MA/data/bt_augmented_agg.csv", sep='\t', index=False, quoting=1)
syn_df.to_csv("~/MA/data/syn_augmented_agg.csv", sep='\t', index=False, quoting=1)
sw_df.to_csv("~/MA/data/sw_augmented_agg.csv", sep='\t', index=False, quoting=1)
del_df.to_csv("~/MA/data/del_augmented_agg.csv", sep='\t', index=False, quoting=1)
print("All done! :)")