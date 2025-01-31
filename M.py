import tensorflow as tf
from tensorflow import keras
import numpy as np
import json
import os

def save_model_components(model, text_vectorizer, lookup_layer, save_dir='saved_model'):
    """Save model components separately"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 1. Save the base model (without text vectorization)
    model.save(f"{save_dir}/base_model")
    
    # 2. Save text vectorizer configuration and vocabulary
    text_vectorizer_config = text_vectorizer.get_config()
    vocab = text_vectorizer.get_vocabulary()
    
    config_data = {
        'config': text_vectorizer_config,
        'vocabulary': vocab
    }
    
    with open(f"{save_dir}/text_vectorizer_config.json", 'w') as f:
        json.dump(config_data, f)
    
    # 3. Save lookup layer vocabulary
    lookup_vocab = lookup_layer.get_vocabulary()
    np.save(f"{save_dir}/lookup_vocab.npy", lookup_vocab)
    
    print(f"Model components saved to {save_dir}")

def load_model_components(save_dir='saved_model'):
    """Load all model components"""
    # 1. Load base model
    model = tf.keras.models.load_model(f"{save_dir}/base_model")
    
    # 2. Load and recreate text vectorizer
    with open(f"{save_dir}/text_vectorizer_config.json", 'r') as f:
        config_data = json.load(f)
    
    text_vectorizer = tf.keras.layers.TextVectorization.from_config(config_data['config'])
    # Adapt vocabulary
    text_vectorizer.set_vocabulary(config_data['vocabulary'])
    
    # 3. Load lookup vocabulary
    lookup_vocab = np.load(f"{save_dir}/lookup_vocab.npy")
    
    return model, text_vectorizer, lookup_vocab

def predict_text(text, save_dir='saved_model', top_k=3):
    """Make predictions using loaded model components"""
    # Load components
    model, text_vectorizer, lookup_vocab = load_model_components(save_dir)
    
    # Prepare input
    if isinstance(text, str):
        text = [text]
    
    # Process text through vectorizer
    vectorized_text = text_vectorizer(text)
    
    # Get predictions
    predictions = model.predict(vectorized_text)
    
    # Process results
    results = []
    for pred in predictions:
        # Get top k predictions
        top_indices = pred.argsort()[-top_k:][::-1]
        
        # Get labels and probabilities
        result = [(lookup_vocab[idx], float(pred[idx])) for idx in top_indices]
        results.append(result)
    
    return results[0] if len(text) == 1 else results

# Example usage:

# 1. After training, save the model components:
"""
save_model_components(
    model=shallow_mlp_model,  # Your trained model
    text_vectorizer=text_vectorizer,
    lookup_layer=lookup,
    save_dir='arxiv_model'
)
"""

# 2. Function to test the saved model
def test_saved_model(save_dir='arxiv_model'):
    """Test the saved model with example texts"""
    try:
        # Test single prediction
        test_text = """This paper presents a novel approach to deep learning 
                      architectures for image classification tasks."""
        
        print("\nTesting single prediction...")
        predictions = predict_text(test_text, save_dir)
        print("\nPredictions for test text:")
        for label, prob in predictions:
            print(f"{label}: {prob:.4f}")
        
        # Test batch prediction
        test_texts = [
            "A new quantum computing algorithm for optimization problems.",
            "Statistical methods for analyzing genomic data."
        ]
        
        print("\nTesting batch prediction...")
        batch_predictions = predict_text(test_texts, save_dir)
        
        print("\nPredictions for multiple texts:")
        for i, preds in enumerate(batch_predictions):
            print(f"\nText {i+1}:")
            for label, prob in preds:
                print(f"{label}: {prob:.4f}")
                
    except Exception as e:
        print(f"Error during prediction: {e}")
        print("\nDebug information:")
        print(f"Model directory exists: {os.path.exists(save_dir)}")
        print(f"Text vectorizer config exists: {os.path.exists(f'{save_dir}/text_vectorizer_config.json')}")
        print(f"Lookup vocabulary exists: {os.path.exists(f'{save_dir}/lookup_vocab.npy')}")
