import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
from sklearn.model_selection import train_test_split
from ast import literal_eval
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load and preprocess data
arxiv_data = pd.read_csv(
    "https://github.com/soumik12345/multi-label-text-classification/releases/download/v0.2/arxiv_data.csv"
)

# Remove duplicates
arxiv_data = arxiv_data[~arxiv_data["titles"].duplicated()]

# Filter rare terms
arxiv_data_filtered = arxiv_data.groupby("terms").filter(lambda x: len(x) > 1)

# Convert string labels to lists
arxiv_data_filtered["terms"] = arxiv_data_filtered["terms"].apply(lambda x: literal_eval(x))

# Split the data
test_split = 0.1
train_df, test_df = train_test_split(
    arxiv_data_filtered,
    test_size=test_split,
    stratify=arxiv_data_filtered["terms"].values,
)
val_df = test_df.sample(frac=0.5)
test_df.drop(val_df.index, inplace=True)

# Create label encoder
terms = tf.ragged.constant(train_df["terms"].values)
lookup = tf.keras.layers.StringLookup(output_mode="multi_hot")
lookup.adapt(terms)
vocab = lookup.get_vocabulary()

# Model parameters
max_sequence_length = 150
vocabulary_size = 50000
embedding_dim = 128
lstm_units = 64
batch_size = 32

# Create text vectorizer
text_vectorizer = layers.TextVectorization(
    max_tokens=vocabulary_size,
    output_mode='int',
    output_sequence_length=max_sequence_length
)
text_vectorizer.adapt(train_df["summaries"].values)

def make_dataset(dataframe, is_train=True):
    """Create a tf.data.Dataset from a DataFrame."""
    labels = tf.ragged.constant(dataframe["terms"].values)
    label_binarized = lookup(labels).numpy()
    
    dataset = tf.data.Dataset.from_tensor_slices(
        (dataframe["summaries"].values, label_binarized)
    )
    
    dataset = dataset.map(
        lambda text, label: (text_vectorizer(text), label),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    if is_train:
        dataset = dataset.shuffle(batch_size * 10)
    
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

# Create datasets
train_dataset = make_dataset(train_df, is_train=True)
val_dataset = make_dataset(val_df, is_train=False)
test_dataset = make_dataset(test_df, is_train=False)

def create_lstm_model():
    """Create the LSTM model architecture."""
    model = keras.Sequential([
        layers.Input(shape=(max_sequence_length,)),
        
        layers.Embedding(
            input_dim=vocabulary_size + 1,  # +1 for padding token
            output_dim=embedding_dim,
            mask_zero=True
        ),
        
        layers.Bidirectional(layers.LSTM(lstm_units, return_sequences=True)),
        layers.Dropout(0.3),
        
        layers.Bidirectional(layers.LSTM(lstm_units)),
        layers.Dropout(0.3),
        
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.4),
        
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        
        layers.Dense(lookup.vocabulary_size(), activation='sigmoid')
    ])
    return model

# Create and compile model
model = create_lstm_model()
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['binary_accuracy']
)

# Add callbacks
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        min_lr=1e-5
    ),
    keras.callbacks.ModelCheckpoint(
        'best_model.h5',
        monitor='val_loss',
        save_best_only=True
    )
]

# Train the model
epochs = 20
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=epochs,
    callbacks=callbacks
)

# Plot training history
def plot_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss plot
    ax1.plot(history.history['loss'], label='Training Loss')
    ax1.plot(history.history['val_loss'], label='Validation Loss')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy plot
    ax2.plot(history.history['binary_accuracy'], label='Training Accuracy')
    ax2.plot(history.history['val_binary_accuracy'], label='Validation Accuracy')
    ax2.set_title('Model Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

plot_history(history)

# Evaluate on test set
test_loss, test_accuracy = model.evaluate(test_dataset)
print(f"\nTest Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# Function to make predictions
def predict_categories(text, model, threshold=0.5):
    """Predict categories for a given text."""
    # Vectorize and predict
    text_vector = text_vectorizer([text])
    predictions = model.predict(text_vector)[0]
    
    # Get categories where prediction exceeds threshold
    predicted_categories = [
        vocab[i] for i, pred in enumerate(predictions) 
        if pred > threshold and vocab[i] != ""
    ]
    
    return predicted_categories, predictions

# Example prediction
sample_text = test_df["summaries"].iloc[0]
predicted_cats, prediction_scores = predict_categories(sample_text, model)
print("\nSample Text:", sample_text[:200], "...\n")
print("Predicted Categories:", predicted_cats)

# Create an inference model that includes preprocessing
inference_model = keras.Sequential([
    text_vectorizer,
    model
])

# Save the inference model
inference_model.save('arxiv_classifier_lstm')

print("\nModel saved successfully. You can load it using:")
print("loaded_model = tf.keras.models.load_model('arxiv_classifier_lstm')")
