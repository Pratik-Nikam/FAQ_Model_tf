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
