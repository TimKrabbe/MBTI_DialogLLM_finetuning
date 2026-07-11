from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, EarlyStoppingCallback
import torch
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig
from huggingface_hub import login
import os

# HF Token login
login(token=os.environ["HF_TOKEN"])

# import data
df = load_dataset("DrinkIcedT/mbti_dialogue")


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


# training args
training_args = SFTConfig(
    output_dir="finetuning/model",
    num_train_epochs=6,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    learning_rate=1e-4,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    logging_steps=10,
    save_strategy="steps",
    save_steps=200,
    save_total_limit=2,
    eval_strategy="steps",
    eval_steps= 200,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    bf16=True,
    fp16=False,
    optim="adamw_torch",
    gradient_checkpointing=True,
    ddp_find_unused_parameters=False,
    report_to="tensorboard",
    max_length=512,
)

# trainer
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=df["train"],
    eval_dataset=df["validation"],
    peft_config=lora_config,
    processing_class=qwen_tokenizer,  # übernimmt Tokenisierung
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
)

trainer.train()
trainer.save_model("finetuning/LoRAadapter")
trainer.push_to_hub("DrinkIcedT/Qwen2.5-7B-Instruct_MBTI-Dialogues_modifiedprompt")
