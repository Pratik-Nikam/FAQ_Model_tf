import faiss
import pickle
import numpy as np
import torch
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from transformers import BertTokenizer, BertModel

class ActionSearchFAQ(Action):
    def name(self):
        return "action_search_faq"

    def __init__(self):
        # Load FAISS index and FAQ data
        self.index = faiss.read_index("faq_index.faiss")
        with open("faq_data.pkl", "rb") as f:
            self.faq_data = pickle.load(f)

        # Load local BERT model and tokenizer
        model_path = "./bert_model"
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model = BertModel.from_pretrained(model_path)
        self.model.eval()

    def get_embedding(self, text):
        """Generate embedding for a given text using local BERT model."""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state[:, 0, :].numpy()  # CLS token embedding

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: dict):
        # Get user input
        user_question = tracker.latest_message['text']

        # Convert user question to embedding
        user_embedding = self.get_embedding(user_question)

        # Search for most similar FAQ
        _, index = self.index.search(user_embedding, 1)  # Top 1 match
        best_match = self.faq_data.iloc[index[0][0]]

        # Send response
        dispatcher.utter_message(text=best_match["answer"])
        return []

####


import yaml
from uuid import uuid4
import os

def generate_rasa_files(qa_pairs, output_dir="data"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Generate responses.yml
    responses = {"version": "3.1", "responses": {}}
    
    faq_examples = []
    faq_response_keys = []

    for idx, pair in enumerate(qa_pairs):
        faq_id = f"faq_{uuid4().hex[:6]}"  # Unique ID for each FAQ
        response_key = f"utter_faq/{faq_id}"

        responses["responses"][response_key] = [{"text": pair["answer"]}]
        
        # Store question and response key for faq.yml
        faq_examples.append(f"- {pair['question']}")
        faq_response_keys.append(f"- {faq_id}")

    # Save responses.yml
    with open(f"{output_dir}/responses.yml", "w") as f:
        yaml.dump(responses, f, default_flow_style=False)

    # Generate faq.yml (retrieval intent with response mapping)
    faq_config = {
        "version": "3.1",
        "nlu": [{
            "intent": "ask_faq",
            "examples": "\n".join(faq_examples),
            "response": "\n".join(faq_response_keys)
        }],
        "retrieval_intents": ["faq"]
    }

    with open(f"{output_dir}/faq.yml", "w") as f:
        yaml.dump(faq_config, f, default_flow_style=False)

# Example Usage
qa_pairs = [
    {"question": "What is the capital of France?", "answer": "The capital of France is Paris."},
    {"question": "What is the population of Paris?", "answer": "The population of Paris is approximately 2.1 million."},
    {"question": "How tall is the Eiffel Tower?", "answer": "The Eiffel Tower is 324 meters tall."}
]

generate_rasa_files(qa_pairs)

###
# Import necessary libraries
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, AdamW
import faiss

# Define a custom dataset
class QADataset(Dataset):
    def __init__(self, qa_pairs, tokenizer, max_length=128):
        self.qa_pairs = qa_pairs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.qa_pairs)

    def __getitem__(self, idx):
        question = self.qa_pairs[idx]['question']
        answer = self.qa_pairs[idx]['answer']
        question_enc = self.tokenizer(
            question, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt'
        )
        answer_enc = self.tokenizer(
            answer, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt'
        )
        return {
            'question_input_ids': question_enc['input_ids'].squeeze(),
            'question_attention_mask': question_enc['attention_mask'].squeeze(),
            'answer_input_ids': answer_enc['input_ids'].squeeze(),
            'answer_attention_mask': answer_enc['attention_mask'].squeeze()
        }

# Load your JSON data
with open('qa_data.json', 'r') as f:
    qa_data = json.load(f)

# Initialize the tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('path_to_your_model')

# Create the dataset and dataloader
dataset = QADataset(qa_data, tokenizer)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Set up the optimizer
optimizer = AdamW(model.parameters(), lr=2e-5)

# Fine-tuning loop
model.train()
for epoch in range(3):  # Number of epochs
    for batch in dataloader:
        optimizer.zero_grad()
        question_outputs = model(
            input_ids=batch['question_input_ids'],
            attention_mask=batch['question_attention_mask']
        )
        answer_outputs = model(
            input_ids=batch['answer_input_ids'],
            attention_mask=batch['answer_attention_mask']
        )
        # Calculate a simple cosine similarity loss
        question_embeddings = question_outputs.last_hidden_state[:, 0, :]  # CLS token
        answer_embeddings = answer_outputs.last_hidden_state[:, 0, :]  # CLS token
        cosine_sim = torch.nn.functional.cosine_similarity(question_embeddings, answer_embeddings)
        loss = 1 - cosine_sim.mean()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch + 1} completed with loss: {loss.item()}")

# Save the fine-tuned model
model.save_pretrained('fine_tuned_bert_model')
tokenizer.save_pretrained('fine_tuned_bert_tokenizer')

# Generate embeddings for FAISS index
model.eval()
question_embeddings = []
answers = []
with torch.no_grad():
    for item in qa_data:
        question = item['question']
        answer = item['answer']
        inputs = tokenizer(question, return_tensors='pt', truncation=True, max_length=128)
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
        question_embeddings.append(embedding)
        answers.append(answer)

# Convert to numpy array
question_embeddings = np.array(question_embeddings).astype('float32')

# Build the FAISS index
dimension = question_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(question_embeddings)

# Save the index and answers
faiss.write_index(index, 'faiss_index.index')
with open('answers.json', 'w') as f:
    json.dump(answers, f)

# Retrieval function
def retrieve_answer(query, k=5):
    inputs = tokenizer(query, return_tensors='pt', truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    query_embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy().astype('float32')
    distances, indices = index.search(np.array([query_embedding]), k)
    return [answers[i] for i in indices[0]]

# Example usage
query = "Your question here"
retrieved_answers = retrieve_answer(query)
print("Retrieved Answers:", retrieved_answers)
