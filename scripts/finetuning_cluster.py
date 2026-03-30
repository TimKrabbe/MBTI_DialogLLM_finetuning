from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig
from huggingface_hub import login
import os

# HF Token login
login(token=os.environ["HF_TOKEN"])

# import data
raw_datasets = load_dataset("DrinkIcedT/mbti_balanced")

raw_train_dataset = raw_datasets["train"]
raw_validation_dataset = raw_datasets["validation"]
raw_test_dataset = raw_datasets["test"]


# model
qwen_checkpoint = "Qwen/Qwen2.5-7B-Instruct"
qwen_tokenizer = AutoTokenizer.from_pretrained(qwen_checkpoint)
qwen_tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    qwen_checkpoint,
    trust_remote_code=True,
    dtype=torch.bfloat16,
)


# LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    task_type="CAUSAL_LM"
)


def convert_to_chatml(example):
    mbti_type = raw_datasets["train"].features["label"].int2str(example["label"])
    prompt = f"Your personality Type is {mbti_type}. What is on your mind?"
    return {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": example["post"]}
        ]
    }

ds = raw_datasets.map(
    convert_to_chatml,
)

#tokenisierung
def tokenize_function(example):
    # Chat Template anwenden
    text = qwen_tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False
    )
    # Tokenisieren
    tokenized = qwen_tokenizer(
        text,
        truncation=True,
        max_length=512,
        padding=False
    )

    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

#Dataset transformieren
tokenized_ds = ds.map(
    tokenize_function,
    remove_columns=ds["train"].column_names # Löscht 'messages', 'post' etc.
)


# training args
training_args = SFTConfig(
    output_dir="cephyr/users/timkra/Alvis/MA/output/model",
    num_train_epochs=6,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    learning_rate=2e-4,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    logging_steps=10,
    save_strategy="steps",
    save_steps=200,
    save_total_limit=3,
    eval_strategy="steps",
    eval_steps= 200,
    load_best_model_at_end=False,
    metric_for_best_model="eval_loss",
    bf16=True,
    fp16=False,
    optim="adamw_torch",
    gradient_checkpointing=True,
    ddp_find_unused_parameters=False,
    report_to="none"
)

# trainer
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds["train"],
    eval_dataset=tokenized_ds["validation"],
    peft_config=lora_config,
)

trainer.train()
trainer.save_model("cephyr/users/timkra/Alvis/MA/output/LoRAadapter")
