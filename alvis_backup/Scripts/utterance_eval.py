from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import torch
import numpy as np
import pandas as pd

df_utterances = pd.read_csv("/cephyr/users/timkra/Alvis/MA/data/df_utterances.csv", sep=';', encoding='utf-8')

# pperplexity
tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
model_gpt2 = GPT2LMHeadModel.from_pretrained('gpt2')
model_gpt2.eval()

def compute_perplexity(text):
    try:
        inputs = tokenizer(text, return_tensors='pt', 
                          truncation=True, max_length=512)
        with torch.no_grad():
            loss = model_gpt2(**inputs, labels=inputs['input_ids']).loss
        return torch.exp(loss).item()
    except:
        return np.nan

# TTR
def compute_ttr(text):
    tokens = text.lower().split()
    if len(tokens) == 0:
        return np.nan
    return len(set(tokens)) / len(tokens)

# Distinct-1/2
def compute_distinct(texts):
    unigrams, bigrams = [], []
    for text in texts:
        tokens = text.lower().split()
        unigrams.extend(tokens)
        bigrams.extend(zip(tokens, tokens[1:]))
    d1 = len(set(unigrams)) / len(unigrams) if unigrams else 0
    d2 = len(set(bigrams)) / len(bigrams) if bigrams else 0
    return d1, d2

#results
df_utterances['perplexity'] = df_utterances['text'].apply(compute_perplexity)
df_utterances['ttr']        = df_utterances['text'].apply(compute_ttr)

print(df_utterances.groupby('model')[['perplexity', 'ttr']].mean())

# Distinct per model
for model_name in ['fine-tuned', 'base']:
    texts = df_utterances[df_utterances['model'] == model_name]['text']
    d1, d2 = compute_distinct(texts)
    print(f"{model_name}: Distinct-1 = {d1:.4f}, Distinct-2 = {d2:.4f}")
