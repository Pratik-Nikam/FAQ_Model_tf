import json
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Save the vocabulary to a JSON file
vocab = text_vectorizer.get_vocabulary()
with open("vocab.json", "w") as f:
    json.dump(vocab, f)

# Save the model (without TextVectorization)
shallow_mlp_model.save("arxiv_text_classifier_model_only")

# Load the vocabulary from the JSON file
with open("vocab.json", "r") as f:
    vocab = json.load(f)

# Load the model (without TextVectorization)
loaded_model = tf.keras.models.load_model("arxiv_text_classifier_model_only")

# Create a tokenizer and set its vocabulary
tokenizer = Tokenizer()
tokenizer.word_index = {word: idx for idx, word in enumerate(vocab)}

# Preprocess raw text
def preprocess_text(texts, max_seqlen=150):
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(sequences, maxlen=max_seqlen, padding="post")
    return padded_sequences

# Example new text data
new_texts = [
    "This paper discusses the application of deep learning in natural language processing.",
    "A new approach to quantum computing using machine learning techniques."
]

# Preprocess the text
preprocessed_text = preprocess_text(new_texts)

# Make predictions
predictions = loaded_model.predict(preprocessed_text)

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
