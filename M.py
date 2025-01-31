import tensorflow as tf
from tensorflow import keras
import numpy as np
from typing import List, Dict

class TextClassifierModel(tf.keras.Model):
    def __init__(self, vectorizer, classifier, label_lookup):
        super().__init__()
        self.vectorizer = vectorizer
        self.classifier = classifier
        self.label_lookup = label_lookup

    def call(self, inputs):
        # Handle both single strings and batches
        if isinstance(inputs, str):
            inputs = [inputs]
        
        # Vectorize the text
        x = self.vectorizer(inputs)
        
        # Get predictions
        predictions = self.classifier(x)
        
        return predictions
    
    def predict_with_labels(self, text: str, threshold: float = 0.5) -> Dict[str, float]:
        """
        Predict with human-readable labels and probabilities
        """
        predictions = self(text)
        
        # Get vocabulary for labels
        vocab = self.label_lookup.get_vocabulary()
        
        # Convert predictions to dictionary of label: probability
        results = {
            vocab[i]: float(prob)
            for i, prob in enumerate(predictions[0])
            if prob >= threshold
        }
        
        # Sort by probability
        return dict(sorted(results.items(), key=lambda x: x[1], reverse=True))

def create_combined_model(text_vectorizer, classifier_model, label_lookup):
    """
    Create a combined model that includes preprocessing and prediction
    """
    return TextClassifierModel(text_vectorizer, classifier_model, label_lookup)

def save_combined_model(model: TextClassifierModel, path: str = "./saved_model"):
    """
    Save the combined model using TensorFlow's SavedModel format
    """
    # Save the model with its custom object
    tf.saved_model.save(model, path)
    print(f"Model saved to {path}")

def load_combined_model(path: str = "./saved_model") -> TextClassifierModel:
    """
    Load the combined model
    """
    model = tf.saved_model.load(path)
    print(f"Model loaded from {path}")
    return model

# Example usage with FastAPI
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class TextRequest(BaseModel):
    text: str
    threshold: float = 0.5

@app.post("/predict")
async def predict(request: TextRequest):
    predictions = model.predict_with_labels(request.text, request.threshold)
    return {"predictions": predictions}

if __name__ == "__main__":
    # Step 1: After training, create and save the combined model
    combined_model = create_combined_model(text_vectorizer, shallow_mlp_model, lookup)
    
    # Example text for testing
    test_text = "This paper introduces a new approach to deep learning..."
    
    # Test prediction before saving
    print("Predictions before saving:")
    print(combined_model.predict_with_labels(test_text))
    
    # Save the model
    save_combined_model(combined_model)
    
    # Load the model
    model = load_combined_model()
    
    # Test prediction after loading
    print("\nPredictions after loading:")
    print(model.predict_with_labels(test_text))
    
    # Start the API server
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

# Example of using the model directly in a script
"""
# Load and use the model
model = load_combined_model("./saved_model")

# Single prediction
text = "This paper presents a novel approach to machine learning..."
predictions = model.predict_with_labels(text, threshold=0.3)
print("\nPredictions for:", text)
for label, prob in predictions.items():
    print(f"{label}: {prob:.4f}")
"""
