from transformers import AutoModelForSequenceClassification, AutoTokenizer
from huggingface_hub import login
import os

############
# in case, upload manually


login=login(os.getenv("HF-TOKEN"))

# load local checkpoints
model = AutoModelForSequenceClassification.from_pretrained("C:/Users/Tim/src/MA/DrinkIcedT/roberta_p/checkpoint-2600")
tokenizer = AutoTokenizer.from_pretrained("DrinkIcedT/roberta-large_MBTI_P")  # tokenizer from hub

# overwrite checkpoint on huggingface
model.push_to_hub("DrinkIcedT/roberta-large_MBTI_p")
tokenizer.push_to_hub("DrinkIcedT/roberta-large_MBTI_P")