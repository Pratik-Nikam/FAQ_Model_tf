import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# Common parameters
max_features = 20000  # Maximum vocabulary size
max_len = 200  # Maximum sequence length
embed_dim = 128  # Embedding dimension

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        return self.layernorm2(out1 + ffn_output)

class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super().__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, stop=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

def create_transformer_model(vocab_size, num_labels):
    # Transformer parameters
    num_heads = 2
    ff_dim = 32
    
    # Input for variable-length sequences of integers
    inputs = layers.Input(shape=(max_len,))
    
    # Embedding layer
    embedding_layer = TokenAndPositionEmbedding(max_len, vocab_size, embed_dim)
    x = embedding_layer(inputs)
    
    # Transformer block
    transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
    x = transformer_block(x)
    
    # Global average pooling
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(20, activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    
    # Output layer with sigmoid for multi-label
    outputs = layers.Dense(num_labels, activation="sigmoid")(x)
    
    return keras.Model(inputs, outputs)

def create_lstm_model(vocab_size, num_labels):
    # Input for variable-length sequences of integers
    inputs = layers.Input(shape=(max_len,))
    
    # Embedding layer
    x = layers.Embedding(vocab_size, embed_dim)(inputs)
    
    # Bidirectional LSTM layers
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(64))(x)
    
    # Dense layers
    x = layers.Dense(20, activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    
    # Output layer with sigmoid for multi-label
    outputs = layers.Dense(num_labels, activation="sigmoid")(x)
    
    return keras.Model(inputs, outputs)

# Data preprocessing function
def preprocess_data(df, tokenizer=None):
    # Prepare text data
    if tokenizer is None:
        tokenizer = keras.preprocessing.text.Tokenizer(num_words=max_features)
        tokenizer.fit_on_texts(df['summaries'])
    
    X = tokenizer.texts_to_sequences(df['summaries'])
    X = keras.preprocessing.sequence.pad_sequences(X, maxlen=max_len)
    
    # Prepare labels
    # Convert string representations of lists to actual lists if needed
    if isinstance(df['terms'].iloc[0], str):
        df['terms'] = df['terms'].apply(eval)
    
    # Create multi-label binarizer
    label_binarizer = keras.preprocessing.text.Tokenizer()
    label_binarizer.fit_on_texts([item for sublist in df['terms'] for item in sublist])
    
    # Convert labels to binary matrix
    y = []
    for terms in df['terms']:
        binary_labels = np.zeros(len(label_binarizer.word_index))
        for term in terms:
            if term in label_binarizer.word_index:
                binary_labels[label_binarizer.word_index[term]-1] = 1
        y.append(binary_labels)
    y = np.array(y)
    
    return X, y, tokenizer, label_binarizer

# Training function
def train_model(model, X_train, y_train, X_val, y_val, epochs=10):
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["binary_accuracy"]
    )
    
    history = model.fit(
        X_train,
        y_train,
        batch_size=32,
        epochs=epochs,
        validation_data=(X_val, y_val)
    )
    
    return history

# Example usage:
"""
# Prepare data
X_train, y_train, tokenizer, label_binarizer = preprocess_data(train_df)
X_val, y_val, _, _ = preprocess_data(val_df, tokenizer)

# Create and train Transformer model
transformer_model = create_transformer_model(max_features, y_train.shape[1])
transformer_history = train_model(transformer_model, X_train, y_train, X_val, y_val)

# Create and train LSTM model
lstm_model = create_lstm_model(max_features, y_train.shape[1])
lstm_history = train_model(lstm_model, X_train, y_train, X_val, y_val)
"""
