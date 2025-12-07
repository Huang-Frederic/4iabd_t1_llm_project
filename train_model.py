import torch
from datasets import Dataset, load_from_disk
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    Trainer, 
    TrainingArguments,
    DataCollatorForLanguageModeling
)

dataset = load_from_disk('imdb_corpus_dataset')
model_name = "distilgpt2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)


tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id

num_params = sum(p.numel() for p in model.parameters())
block_size = 128

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=block_size,
    )

tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"],
    desc="Tokenisation"
)
tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = tokenized_dataset["train"]
eval_dataset = tokenized_dataset["test"]


output_dir = "./distilgpt2-imdb-finetuned"

training_args = TrainingArguments(
    output_dir=output_dir,
    eval_strategy="epoch",  
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
)


data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
)

trainer.train()

trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)

eval_results = trainer.evaluate()
print(eval_results)

dataset = load_from_disk('imdb_instructions_dataset')
model_name = "distilgpt2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)


tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id

num_params = sum(p.numel() for p in model.parameters())
block_size = 128

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=block_size,
    )

tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"],
    desc="Tokenisation"
)

print(f"Tokenized {len(tokenized_dataset)} items")

tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = tokenized_dataset["train"]
eval_dataset = tokenized_dataset["test"]


output_dir = "./distilgpt2-imdb-instructions-finetuned"

training_args = TrainingArguments(
    output_dir=output_dir,
    eval_strategy="epoch",  
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
)

trainer.train()


trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)

eval_results = trainer.evaluate()
print(eval_results)

from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType

dataset = load_from_disk("imdb_instructions_dataset")

model_name = "gpt2-medium"  
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["c_attn", "c_proj"], 
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()  

block_size = 256

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=block_size,
    )

tokenized = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
tokenized = tokenized.train_test_split(test_size=0.1, seed=42)
train_dataset = tokenized["train"]
eval_dataset = tokenized["test"]

training_args = TrainingArguments(
    output_dir="./gpt2-imdb-lora",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-4,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    logging_steps=10,
)

from transformers import DataCollatorForLanguageModeling
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
)

trainer.train()
trainer.save_model("./gpt2-imdb-lora")
tokenizer.save_pretrained("./gpt2-imdb-lora")
