

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Set pad_token to eos_token
tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
    return tokenizer(examples["input_text"], truncation=True, padding="max_length", max_length=512)

def data_collator(features):
    batch = tokenizer.pad(
        features,
        padding=True,
        max_length=512,
        return_tensors="pt"
    )
    return batch
