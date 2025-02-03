import tensorflow as tf
from tensorflow.keras import layers, models
import pandas as pd
import numpy as np
import pickle

# Load and prepare data
arxiv_data = pd.read_csv("arxiv_data.csv")
arxiv_data_filtered = arxiv_data[~arxiv_data["titles"].duplicated()]
arxiv_data_filtered["terms"] = arxiv_data_filtered["terms"].apply(literal_eval)

# Train/validation split
train_df, val_df = train_test_split(
    arxiv_data_filtered,
    test_size=0.1,
    stratify=arxiv_data_filtered["terms"].values
)

# -------------------------------------------------
# Critical Fix: Proper Layer Initialization
# -------------------------------------------------
# 1. Initialize and adapt StringLookup
label_terms = tf.ragged.constant(train_df["terms"].values)
label_lookup = layers.StringLookup(output_mode="multi_hot")
label_lookup.adapt(label_terms)
label_vocab = label_lookup.get_vocabulary()

# 2. Initialize and adapt TextVectorization
text_vectorizer = layers.TextVectorization(
    max_tokens=20000,
    ngrams=2,
    output_mode="tf_idf",
    name="text_vectorizer"
)

# Adapt on text data
text_ds = tf.data.Dataset.from_tensor_slices(
    train_df["summaries"].values
).batch(128)
with tf.device("/CPU:0"):
    text_vectorizer.adapt(text_ds)

# -------------------------------------------------
# Build End-to-End Model
# -------------------------------------------------
inputs = tf.keras.Input(shape=(1,), dtype=tf.string, name="text_input")
x = text_vectorizer(inputs)
x = layers.Dense(512, activation="relu")(x)
x = layers.Dense(256, activation="relu")(x)
outputs = layers.Dense(len(label_vocab), activation="sigmoid")(x)
model = tf.keras.Model(inputs, outputs)

# Compile and train
model.compile(
    loss="binary_crossentropy",
    optimizer="adam",
    metrics=["binary_accuracy"]
)
model.fit(
    text_ds.map(lambda x: (x, label_lookup(tf.ragged.constant(train_df["terms"].values)))),
    epochs=2
)

# -------------------------------------------------
# Save with Custom Object Handling
# -------------------------------------------------
# Save model weights and config separately
model.save_weights("model_weights.weights.h5")
with open("model_config.pkl", "wb") as f:
    pickle.dump({
        "text_vectorizer": text_vectorizer,
        "label_lookup": label_lookup
    }, f)

# Save vocabulary
with open("label_vocab.pkl", "wb") as f:
    pickle.dump(label_vocab, f)


# Load components
with open("model_config.pkl", "rb") as f:
    config = pickle.load(f)

loaded_model = models.Model.from_config(
    model.get_config(),
    custom_objects={
        "TextVectorization": layers.TextVectorization,
        "StringLookup": layers.StringLookup
    }
)
loaded_model.load_weights("model_weights.weights.h5")

# Restore layer states
loaded_model.get_layer("text_vectorizer").adapt(text_ds)
loaded_model.get_layer("string_lookup").adapt(label_terms)

# Load vocabulary
with open("label_vocab.pkl", "rb") as f:
    label_vocab = pickle.load(f)


def predict(texts, threshold=0.5):
    # Convert to tensor with shape (batch_size, 1)
    input_tensor = tf.constant(texts, dtype=tf.string)[:, tf.newaxis]
    
    # Predict
    preds = loaded_model.predict(input_tensor)
    
    # Get labels
    return [
        [label_vocab[i] for i in np.where(pred > threshold)[0]]
        for pred in preds
    ]

# Usage
new_texts = ["Your research abstract here..."]
print(predict(new_texts))


