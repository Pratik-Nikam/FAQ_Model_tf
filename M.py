import tensorflow as tf
from tensorflow import keras
import numpy as np

# 1. Create a custom model class that includes everything
class TextClassifier(keras.Model):
    def __init__(self, text_vectorizer, base_model, lookup_layer, **kwargs):
        super().__init__(**kwargs)
        self.text_vectorizer = text_vectorizer
        self.base_model = base_model
        self.lookup_layer = lookup_layer
        
    def call(self, inputs):
        # Process raw text through the pipeline
        x = self.text_vectorizer(inputs)
        return self.base_model(x)
    
    def get_vocabulary(self):
        return self.lookup_layer.get_vocabulary()

# 2. Function to create and save the complete model
def create_complete_model(text_vectorizer, shallow_mlp_model, lookup_layer):
    """Combine all components into a single model"""
    complete_model = TextClassifier(
        text_vectorizer=text_vectorizer,
        base_model=shallow_mlp_model,
        lookup_layer=lookup_layer
    )
    
    # Create input signature for the model
    complete_model.compile(
        loss="binary_crossentropy",
        optimizer="adam",
        metrics=["binary_accuracy"]
    )
    
    # Build the model with sample input
    sample_text = ["Sample text to build model"]
    _ = complete_model(sample_text)
    
    return complete_model

# 3. Save the model after training is complete
"""
# After training your original model:
complete_model = create_complete_model(text_vectorizer, shallow_mlp_model, lookup)
complete_model.save('complete_model')
"""

# 4. Function to load and use the model
def predict_text(text, model_path='complete_model', top_k=3):
    """
    Load model and predict categories for new text
    
    Args:
        text: String or list of strings to classify
        model_path: Path to saved model
        top_k: Number of top predictions to return
    """
    # Load the complete model
    model = tf.keras.models.load_model(model_path)
    
    # Ensure input is in the correct format
    if isinstance(text, str):
        text = [text]
    
    # Make prediction
    predictions = model.predict(text)
    
    # Get vocabulary for labels
    vocab = model.get_vocabulary()
    
    # Process results
    results = []
    for pred in predictions:
        # Get top k predictions
        top_indices = pred.argsort()[-top_k:][::-1]
        # Get labels and probabilities
        result = [(vocab[idx], float(pred[idx])) for idx in top_indices]
        results.append(result)
    
    return results[0] if len(text) == 1 else results

# Example usage:
"""
# 1. After training, create and save the complete model:
complete_model = create_complete_model(text_vectorizer, shallow_mlp_model, lookup)
complete_model.save('complete_model')

# 2. Later, to use the model:
text = "This is a research paper about deep learning."
predictions = predict_text(text, model_path='complete_model')
for label, prob in predictions:
    print(f"{label}: {prob:.4f}")
"""

# Test function to verify everything works
def test_model(model_path='complete_model'):
    """Test the saved model with example texts"""
    try:
        # Test single prediction
        test_text = "This paper presents a novel approach to deep learning."
        print("\nTesting single prediction...")
        predictions = predict_text(test_text, model_path)
        print("\nPredictions for single text:")
        for label, prob in predictions:
            print(f"{label}: {prob:.4f}")
        
        # Test multiple predictions
        test_texts = [
            "A new quantum computing algorithm.",
            "Statistical methods in machine learning."
        ]
        print("\nTesting multiple predictions...")
        batch_predictions = predict_text(test_texts, model_path)
        print("\nPredictions for multiple texts:")
        for i, preds in enumerate(batch_predictions):
            print(f"\nText {i+1}:")
            for label, prob in preds:
                print(f"{label}: {prob:.4f}")
                
    except Exception as e:
        print(f"Error during testing: {e}")
