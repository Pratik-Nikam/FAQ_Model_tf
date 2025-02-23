import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("data.csv")
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
val_df, test_df = train_test_split(test_df, test_size=0.5, random_state=42)

all_labels = set()
for labels in df["labels"]:
    all_labels.update(labels)
all_labels = list(all_labels)
label_to_id = {label: i for i, label in enumerate(all_labels)}
id_to_label = {i: label for i, label in enumerate(all_labels)}
num_labels = len(all_labels)

def labels_to_ids(labels):
    return [label_to_id[label] for label in labels]

train_df["label_ids"] = train_df["labels"].apply(labels_to_ids)
val_df["label_ids"] = val_df["labels"].apply(labels_to_ids)
test_df["label_ids"] = test_df["labels"].apply(labels_to_ids)

from transformers import BertTokenizer, BertForSequenceClassification

tokenizer = BertTokenizer.from_pretrain("bert-base-uncased")
model = BertForSequenceClassification.from_pretrain("bert-base-uncased", num_labels=num_labels, problem_type="multi_label_classification")

from datasets import Dataset

train_dataset = Dataset.from_pandas(train_df[["summary", "label_ids"]])
val_dataset = Dataset.from_pandas(val_df[["summary", "label_ids"]])
test_dataset = Dataset.from_pandas(test_df[["summary", "label_ids"]])

def tokenize_function(examples):
    tokenized_summaries = tokenizer(examples["summary"], padding="max_length", truncation=True, max_length=512)
    labels = []
    for label_ids in examples["label_ids"]:
        binary_label = [0] * num_labels
        for label_id in label_ids:
            binary_label[label_id] = 1
        labels.append(binary_label)
    tokenized_summaries["labels"] = labels
    return tokenized_summaries

train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
val_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

from transformers import TrainingArguments, Trainer
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

training_args = TrainingArguments(
    output_dir="bert_multi_label",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="logs"
)

def compute_metrics(p):
    pred_labels = np.round(p.predictions)
    true_labels = p.label_ids
    f1 = f1_score(true_labels, pred_labels, average="micro")
    precision = precision_score(true_labels, pred_labels, average="micro")
    recall = recall_score(true_labels, pred_labels, average="micro")
    return {"f1": f1, "precision": precision, "recall": recall}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

trainer.train()

trainer.evaluate(test_dataset)
def predict_labels(model, tokenizer, summary):
    inputs = tokenizer(summary, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    probabilities = torch.sigmoid(outputs.logits).detach().cpu().numpy()
    predicted_labels = [all_labels[i] for i, prob in enumerate(probabilities[0]) if prob > 0.5]
    return predicted_labels



