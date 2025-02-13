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


# Calculate class weights
class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
class_weights = dict(enumerate(class_weights))

# Add to model.fit()
history = shallow_mlp_model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=epochs,
    class_weight=class_weights
    
    from tensorflow.keras.models import load_model, Sequential
 model_for_inference.save('/tmp/keras-model.h5')
 load_ = load_model('/tmp/keras-model.h5')
load_.compile(
    loss="binary_crossentropy", optimizer="adam", metrics=["binary_accuracy"]

The extracted text appears to be partially related to banking and lockbox processing, which might not be relevant to your model development request. However, I'll structure your model approval form with clear requirements based on your project description.


---

Model Approval Form

1. Model Overview

We propose developing two machine learning models for multi-class classification using Keras and TensorFlow:

Model 1: Classifies support tickets into predefined categories based on the ticket description and associated metadata.

Model 2: Utilizes case metadata and support ticket information to identify issues and provide guided resolutions. This model will leverage BERT or SBERT for better sentence understanding.



---

2. Model Requirements

Objective: Automate ticket classification and issue resolution guidance.

Type: Multi-class classification.

Techniques Used:

Model 1: Neural Networks (Deep Learning)

Model 2: Transformer-based Models (BERT/SBERT)


Evaluation Metrics: Accuracy, Precision, Recall, F1-Score.



---

3. Data Requirements

Input Data:

Support Tickets (text description, categories/flavors, metadata).

Case Metadata (past issue resolutions, ticket history).


Data Format: CSV, JSON, or relational database.

Data Preprocessing:

Text Cleaning (stopword removal, tokenization).

Embedding Generation (TF-IDF, Word2Vec, or BERT embeddings).

Label Encoding for classification.




---

4. Technical Requirements

Programming Language: Python

Libraries: Keras, TensorFlow, Hugging Face Transformers, Pandas, NumPy, Scikit-Learn.

Infrastructure: GPU support for BERT-based models (Cloud or Local Setup).

Model Training: Supervised learning with labeled historical data.

Deployment: Flask/FastAPI for API integration or embedding in an existing support system.



---

5. Expected Outcomes

Automated Ticket Classification: Reducing manual effort in categorizing support tickets.

Guided Issue Resolution: AI-driven suggestions based on historical data.

Improved Response Time: Faster resolution through automated recommendations.



---

Would you like me to refine or add any specific details based on your organization's requirements?


    
