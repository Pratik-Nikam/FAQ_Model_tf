import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd

# Data preprocessing function
def preprocess_data(df, text_col, label_col, max_seq_length=128, vocab_size=10000):
    # Convert labels to numerical format
    label_lookup = layers.StringLookup(output_mode="multi_hot")
    label_lookup.adapt(tf.ragged.constant(df[label_col].values))
    
    # Create text vectorization layer
    text_vectorizer = layers.TextVectorization(
        max_tokens=vocab_size,
        output_sequence_length=max_seq_length,
        output_mode='int'
    )
    text_vectorizer.adapt(df[text_col].values)
    
    return text_vectorizer, label_lookup

# LSTM Model
def create_lstm_model(vocab_size, num_labels, embedding_dim=128, lstm_units=64):
    model = models.Sequential([
        layers.Embedding(vocab_size, embedding_dim),
        layers.Bidirectional(layers.LSTM(lstm_units, return_sequences=True)),
        layers.Bidirectional(layers.LSTM(lstm_units)),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_labels, activation='sigmoid')
    ])
    return model

# Transformer Block
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = models.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

# Transformer Model
def create_transformer_model(vocab_size, num_labels, maxlen=128, embed_dim=128, num_heads=8, ff_dim=128):
    inputs = layers.Input(shape=(maxlen,))
    embedding_layer = layers.Embedding(vocab_size, embed_dim)
    x = embedding_layer(inputs)
    transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
    x = transformer_block(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(64, activation="relu")(x)
    outputs = layers.Dense(num_labels, activation="sigmoid")(x)
    model = models.Model(inputs=inputs, outputs=outputs)
    return model

# Training function
def train_model(model, train_dataset, val_dataset, epochs=10):
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['binary_accuracy']
    )
    
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        batch_size=32
    )
    return history

# Create tf.data.Dataset
def create_dataset(texts, labels, batch_size=32):
    dataset = tf.data.Dataset.from_tensor_slices((texts, labels))
    dataset = dataset.shuffle(batch_size * 10).batch(batch_size)
    return dataset.prefetch(tf.data.AUTOTUNE)

# Main execution function
def run_classification(df, text_col='Description', label_col='Flavors'):
    # Preprocess data
    text_vectorizer, label_lookup = preprocess_data(df, text_col, label_col)
    
    # Split data
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    val_df = test_df.sample(frac=0.5)
    test_df = test_df.drop(val_df.index)
    
    # Create datasets
    def prepare_data(texts, labels):
        vectorized_texts = text_vectorizer(texts)
        encoded_labels = label_lookup(labels)
        return vectorized_texts, encoded_labels
    
    train_dataset = create_dataset(*prepare_data(train_df[text_col], train_df[label_col]))
    val_dataset = create_dataset(*prepare_data(val_df[text_col], val_df[label_col]))
    test_dataset = create_dataset(*prepare_data(test_df[text_col], test_df[label_col]))
    
    # Create and train LSTM model
    lstm_model = create_lstm_model(
        vocab_size=len(text_vectorizer.get_vocabulary()),
        num_labels=len(label_lookup.get_vocabulary())
    )
    print("Training LSTM model...")
    lstm_history = train_model(lstm_model, train_dataset, val_dataset)
    
    # Create and train Transformer model
    transformer_model = create_transformer_model(
        vocab_size=len(text_vectorizer.get_vocabulary()),
        num_labels=len(label_lookup.get_vocabulary())
    )
    print("Training Transformer model...")
    transformer_history = train_model(transformer_model, train_dataset, val_dataset)
    
    return {
        'lstm_model': lstm_model,
        'transformer_model': transformer_model,
        'text_vectorizer': text_vectorizer,
        'label_lookup': label_lookup,
        'lstm_history': lstm_history,
        'transformer_history': transformer_history
    }

# Example usage
if __name__ == "__main__":
    # Load and prepare your data
    df = pd.read_csv('your_data.csv')
    
    # First task: Predict flavors from description
    results = run_classification(df, text_col='Description', label_col='Flavors')
    
    # Second task: Predict resolution flavor from resolution
    resolution_results = run_classification(df, text_col='Resolution', label_col='Resolution_Flavor')
    
    # Third task: Predict team
    team_results = run_classification(df, text_col='Description', label_col='Team')
