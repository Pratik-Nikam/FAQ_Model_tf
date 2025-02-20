import tensorflow as tf
import numpy as np

# Save the entire model (including TextVectorization)
model_for_inference.save("arxiv_text_classifier_full")

# Load the saved model
loaded_model = tf.keras.models.load_model("arxiv_text_classifier_full")

# Access the vocabulary from the TextVectorization layer
text_vectorization_layer = loaded_model.layers[0]
vocab = text_vectorization_layer.get_vocabulary()

# Example new text data
new_texts = [
    "This paper discusses the application of deep learning in natural language processing.",
    "A new approach to quantum computing using machine learning techniques."
]

# Make predictions
predictions = loaded_model.predict(new_texts)

# Decode the predictions
def decode_predictions(predicted_probabilities, top_k=3):
    top_k_labels = []
    for proba in predicted_probabilities:
        top_k_indices = np.argsort(proba)[-top_k:][::-1]
        top_k_labels.append([vocab[i] for i in top_k_indices])
    return top_k_labels

# Get the top-k predicted labels
top_k_predictions = decode_predictions(predictions, top_k=3)

# Print the results
for i, text in enumerate(new_texts):
    print(f"Text: {text}")
    print(f"Predicted Labels: {', '.join(top_k_predictions[i])}")
    print(" ")
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset
from sklearn.preprocessing import LabelEncoder
from sentence_transformers import SentenceTransformer, util

# Load FAQ Dataset (Ensure CSV has "question" and "answer" columns)
df = pd.read_csv("faq_data.csv")

# Encode answers as labels
label_encoder = LabelEncoder()
df["label"] = label_encoder.fit_transform(df["answer"])

# Convert to Hugging Face Dataset
dataset = Dataset.from_pandas(df)

# Load tokenizer from MiniLM v6
tokenizer = AutoTokenizer.from_pretrained("Bert/minilm2_v6")

# Tokenization function
def tokenize_function(examples):
    return tokenizer(examples["question"], padding="max_length", truncation=True)

# Apply tokenization
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Load MiniLM model for classification
num_labels = df["label"].nunique()
model = AutoModelForSequenceClassification.from_pretrained("Bert/minilm2_v6", num_labels=num_labels)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
)

# Train the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained("fine_tuned_miniLM_faq")
tokenizer.save_pretrained("fine_tuned_miniLM_faq")

# Load MiniLM Sentence Transformer for retrieval-based search
retrieval_model = SentenceTransformer("Bert/minilm2_v6")
faq_embeddings = retrieval_model.encode(df["question"].tolist(), convert_to_tensor=True)

# Function to get answer from fine-tuned model
def get_answer(question):
    inputs = tokenizer(question, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    predicted_label = outputs.logits.argmax().item()
    return label_encoder.inverse_transform([predicted_label])[0]

# Function to get best answer using retrieval-based search
def find_best_answer(user_query):
    query_embedding = retrieval_model.encode(user_query, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(query_embedding, faq_embeddings)
    best_match_idx = scores.argmax()
    return df.iloc[best_match_idx]["answer"]

# Example usage
if __name__ == "__main__":
    user_question = "How can I reset my password?"
    print("Fine-tuned model's answer:", get_answer(user_question))
    print("Retrieval-based answer:", find_best_answer(user_question))
