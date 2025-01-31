import tensorflow as tf
from tensorflow import keras
import numpy as np
from typing import List
import json

# First, save both the text vectorizer and the model
def save_model(model, text_vectorizer, base_path: str = "./saved_model"):
    # Save the text vectorizer configuration and vocabulary
    vectorizer_config = text_vectorizer.get_config()
    vocab = text_vectorizer.get_vocabulary()
    
    config_path = f"{base_path}/vectorizer_config.json"
    vocab_path = f"{base_path}/vectorizer_vocab.json"
    
    with open(config_path, 'w') as f:
        json.dump(vectorizer_config, f)
    
    with open(vocab_path, 'w') as f:
        json.dump(vocab, f)
    
    # Save the model
    model.save(f"{base_path}/model")
    print(f"Model and vectorizer saved to {base_path}")

# Load the saved model and text vectorizer
def load_model(base_path: str = "./saved_model"):
    # Load the text vectorizer configuration and vocabulary
    with open(f"{base_path}/vectorizer_config.json", 'r') as f:
        vectorizer_config = json.load(f)
    
    with open(f"{base_path}/vectorizer_vocab.json", 'r') as f:
        vocab = json.load(f)
    
    # Recreate the text vectorizer
    text_vectorizer = tf.keras.layers.TextVectorization.from_config(vectorizer_config)
    text_vectorizer.set_vocabulary(vocab)
    
    # Load the model
    model = keras.models.load_model(f"{base_path}/model")
    
    return model, text_vectorizer

# Create a classifier class for easy inference
class TextClassifier:
    def __init__(self, model, text_vectorizer, label_lookup):
        self.model = model
        self.text_vectorizer = text_vectorizer
        self.label_lookup = label_lookup
        self.vocab = label_lookup.get_vocabulary()
    
    def predict(self, text: str, top_k: int = 3) -> List[tuple]:
        # Preprocess and predict
        vectorized_text = self.text_vectorizer([text])
        predictions = self.model.predict(vectorized_text)
        
        # Get top k predictions
        top_k_indices = predictions[0].argsort()[-top_k:][::-1]
        top_k_probs = predictions[0][top_k_indices]
        
        # Convert to labels
        results = [(self.vocab[idx], float(prob)) 
                  for idx, prob in zip(top_k_indices, top_k_probs)]
        
        return results

# Example FastAPI implementation
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class TextRequest(BaseModel):
    text: str
    top_k: int = 3

class PredictionResponse(BaseModel):
    labels: List[tuple]

@app.post("/predict")
async def predict(request: TextRequest):
    predictions = classifier.predict(request.text, request.top_k)
    return {"labels": predictions}

# Usage example
if __name__ == "__main__":
    # Save the model (run this after training)
    save_model(shallow_mlp_model, text_vectorizer)
    
    # Load the model (run this when deploying)
    loaded_model, loaded_vectorizer = load_model()
    
    # Create classifier instance
    classifier = TextClassifier(loaded_model, loaded_vectorizer, lookup)
    
    # Example prediction
    text = "This paper presents a novel approach to quantum computing..."
    predictions = classifier.predict(text)
    print(f"Text: {text}")
    print("Predictions:")
    for label, prob in predictions:
        print(f"- {label}: {prob:.4f}")
    
    # Start the API server
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
