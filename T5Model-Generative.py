# T5 Model

import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_excel("data.xlsx")
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
val_df, test_df = train_test_split(test_df, test_size=0.5, random_state=42)

from datasets import Dataset
train_data_category = [{"input": row["Description"], "output": row["Category"]} for _, row in train_df.iterrows()]
val_data_category = [{"input": row["Description"], "output": row["Category"]} for _, row in val_df.iterrows()]
train_dataset_category = Dataset.from_list(train_data_category)
val_dataset_category = Dataset.from_list(val_data_category)

# T5 Small

from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments

model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrain(model_name)
model_category = T5ForConditionalGeneration.from_pretrain(model_name)

def tokenize_function(examples):
    inputs = tokenizer(examples["input"], padding="max_length", truncation=True, max_length=512)
    outputs = tokenizer(examples["output"], padding="max_length", truncation=True, max_length=512)
    return {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"], "labels": outputs["input_ids"]}

train_dataset_category = train_dataset_category.map(tokenize_function, batched=True)
val_dataset_category = val_dataset_category.map(tokenize_function, batched=True)
train_dataset_category.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
val_dataset_category.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

training_args = TrainingArguments(
    output_dir="t5_category_classification",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="logs"
)

trainer = Trainer(
    model=model_category,
    args=training_args,
    train_dataset=train_dataset_category,
    eval_dataset=val_dataset_category
)
trainer.train()
trainer.save_model("t5_category_classification")

descriptions_by_category = df.groupby("Category")["Description"].apply(list).to_dict()

import torch
from torch import tensor

def get_embedding(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(input_ids=inputs["input_ids"], decoder_input_ids=None, return_dict=True)
    last_hidden_state = outputs.encoder_last_hidden_state
    embedding = torch.mean(last_hidden_state, dim=1)
    return embedding

embeddings_by_category = {}
for category, descriptions in descriptions_by_category.items():
    embeddings = []
    for desc in descriptions:
        embedding = get_embedding(model_category, tokenizer, desc)
        embeddings.append((desc, embedding))
    embeddings_by_category[category] = embeddings

def classify_question(model, tokenizer, question):
    inputs = tokenizer(question, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model.generate(inputs["input_ids"], max_length=50, num_return_sequences=1)
    category = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return category

def cosine_similarity(embedding1, embedding2):
    dot_product = torch.dot(embedding1, embedding2)
    norm1 = torch.norm(embedding1)
    norm2 = torch.norm(embedding2)
    similarity = dot_product / (norm1 * norm2)
    return similarity

def find_most_similar_description(category, user_embedding, embeddings_by_category):
    descriptions = embeddings_by_category[category]
    max_similarity = -1
    best_description = None
    for desc, desc_embedding in descriptions:
        similarity = cosine_similarity(user_embedding, desc_embedding)
        if similarity > max_similarity:
            max_similarity = similarity
            best_description = desc
    return best_description

# Resolution Generation
train_data_resolution = [{"input": f"user's question: {row['Description']} description: {row['Description']}", "output": row["Resolution"]} for _, row in train_df.iterrows()]
val_data_resolution = [{"input": f"user's question: {row['Description']} description: {row['Description']}", "output": row["Resolution"]} for _, row in val_df.iterrows()]

train_dataset_resolution = Dataset.from_list(train_data_resolution)
val_dataset_resolution = Dataset.from_list(val_data_resolution)
train_dataset_resolution = train_dataset_resolution.map(tokenize_function, batched=True)
val_dataset_resolution = val_dataset_resolution.map(tokenize_function, batched=True)
train_dataset_resolution.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
val_dataset_resolution.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

model_resolution = T5ForConditionalGeneration.from_pretrain(model_name)
trainer = Trainer(
    model=model_resolution,
    args=training_args,
    train_dataset=train_dataset_resolution,
    eval_dataset=val_dataset_resolution
)
trainer.train()
trainer.save_model("t5_resolution_generation")
def get_resolution(user_question):
    category = classify_question(model_category, tokenizer, user_question)
    user_embedding = get_embedding(model_category, tokenizer, user_question)
    retrieved_description = find_most_similar_description(category, user_embedding, embeddings_by_category)
    input_text = f"user's question: {user_question} description: {retrieved_description}"
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model_resolution.generate(inputs["input_ids"], max_length=100, num_return_sequences=1)
    resolution = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return resolution



