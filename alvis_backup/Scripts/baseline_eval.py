from trl import SFTTrainer, SFTConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, EarlyStoppingCallback
from datasets import load_dataset
import torch
from huggingface_hub import login
import os

# HF Token login
login(token=os.environ["HF_TOKEN"])

df = load_dataset("DrinkIcedT/mbti_dialogue")
eval_df = df["validation"]

model_id = "Qwen/Qwen2.5-7B-Instruct"
qwen_tokenizer = AutoTokenizer.from_pretrained(model_id)
qwen_tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    dtype=torch.bfloat16,
)

trainer = SFTTrainer(
    model=model,
    args=SFTConfig(
        output_dir="finetuning/baseline_eval_only",
        per_device_eval_batch_size=4,
        max_length=512,
        bf16=True,
        do_train=False,
        do_eval=True,
    ),
    train_dataset=eval_df,
    eval_dataset=eval_df,
    processing_class = qwen_tokenizer,
)

results = trainer.evaluate()
print(results)
