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

