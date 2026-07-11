from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel
from huggingface_hub import login
import pandas as pd
import random
import torch
import os

rseed = 42

login(token=os.environ["HF_TOKEN"])

adapter = "DrinkIcedT/Qwen2.5-7B-Instruct_MBTI-Dialogues_modifiedprompt"
qwen_base = "Qwen/Qwen2.5-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(qwen_base)
tokenizer.pad_token = tokenizer.eos_token 
tokenizer.padding_side = "left"

# Base Model
model_base = AutoModelForCausalLM.from_pretrained(
    qwen_base,
    trust_remote_code=True,
    dtype=torch.bfloat16,
)

mbti_labels = ["ENFJ", "ENFP", "ENTJ", "ENTP", "ESFJ", "ESFP", "ESTJ", "ESTP", 
               "INFJ", "INFP", "INTJ", "INTP", "ISFJ", "ISFP", "ISTJ", "ISTP"]

random.seed(rseed)
#n_rows  = 800
#labels = [random.choice(mbti_labels) for _ in range(n_rows)]
labels = mbti_labels * 50
random.shuffle(labels)
prompts = [f"<|im_start|>user\nYour personality Type is {l}. Share your thoughts on a topic that's been on your mind recently. Do not reference personality types, traits, or psychological concepts — express your personality through your writing style and perspective, not by describing it.<|im_end|>\n<|im_start|>assistant\n" for l in labels]

generation_kwargs = {
    "max_new_tokens": 150,
    "min_new_tokens": 100,
    "return_full_text": False
}

#### base text generation
pipe_base = pipeline(
    task="text-generation",
    model=model_base,
    tokenizer=tokenizer,
    batch_size=8 
)

print("Generating Base Texts...")
# Wir nutzen die Pipeline als Generator für Effizienz
base_outputs = pipe_base(prompts, **generation_kwargs)
texts_base = [out[0]['generated_text'] for out in base_outputs]

del pipe_base, model_base
torch.cuda.empty_cache()


# Tuned Model
model_tuned_base = AutoModelForCausalLM.from_pretrained(
    qwen_base,
    trust_remote_code=True,
    dtype=torch.bfloat16,
)
model_tuned = PeftModel.from_pretrained(model_tuned_base, adapter) # lora adapter


#### text generation tuned model
pipe_tuned = pipeline(
    task="text-generation",
    model=model_tuned,
    tokenizer=tokenizer,
    batch_size=8
)

print("Generating Tuned Texts...")
tuned_outputs = pipe_tuned(prompts, **generation_kwargs)
texts_tuned = [out[0]['generated_text'] for out in tuned_outputs]

def mask_mbti_label(text, label):
    return text.replace(label, "<mbti>")

texts_base_masked = [mask_mbti_label(t, l) for t, l in zip(texts_base, labels)]
texts_tuned_masked = [mask_mbti_label(t, l) for t, l in zip(texts_tuned, labels)]

df_base = pd.DataFrame({"labels": labels, "prompt": prompts, "text": texts_base_masked})
df_tuned = pd.DataFrame({"labels": labels, "prompt": prompts, "text": texts_tuned_masked})



df_base.to_csv("/cephyr/users/timkra/Alvis/MA/data/base_test_modifiedprompt_10.csv", index=False)
df_tuned.to_csv("/cephyr/users/timkra/Alvis/MA/data/tuned_test_modifiedprompt_10.csv", index=False)

print("Success!")
