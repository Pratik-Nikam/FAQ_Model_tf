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
