import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd

class DualInputTextClassifier:
    def __init__(self, vocab_size=10000, max_length=128, embedding_dim=128):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.embedding_dim = embedding_dim
        
        # Text vectorization layers for both inputs
        self.description_vectorizer = layers.TextVectorization(
            max_tokens=vocab_size,
            output_sequence_length=max_length,
            output_mode='int'
        )
        
        self.resolution_vectorizer = layers.TextVectorization(
            max_tokens=vocab_size,
            output_sequence_length=max_length,
            output_mode='int'
        )
        
        # Label encoding layer
        self.label_lookup = layers.StringLookup(output_mode="multi_hot")
    
    def adapt_data(self, descriptions, resolutions, labels):
        """Adapt the vectorization layers to the data"""
        self.description_vectorizer.adapt(descriptions)
        self.resolution_vectorizer.adapt(resolutions)
        self.label_lookup.adapt(tf.ragged.constant(labels))
        
        self.num_labels = len(self.label_lookup.get_vocabulary())
    
    def create_lstm_model(self):
        """Create a dual-input LSTM model"""
        # Input layers
        description_input = layers.Input(shape=(self.max_length,), name='description_input')
        resolution_input = layers.Input(shape=(self.max_length,), name='resolution_input')
        
        # Shared embedding layer
        embedding = layers.Embedding(self.vocab_size, self.embedding_dim)
        
        # Process description path
        desc_embedded = embedding(description_input)
        desc_lstm = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(desc_embedded)
        desc_lstm = layers.Bidirectional(layers.LSTM(32))(desc_lstm)
        desc_dense = layers.Dense(64, activation='relu')(desc_lstm)
        
        # Process resolution path
        res_embedded = embedding(resolution_input)
        res_lstm = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(res_embedded)
        res_lstm = layers.Bidirectional(layers.LSTM(32))(res_lstm)
        res_dense = layers.Dense(64, activation='relu')(res_lstm)
        
        # Combine both paths
        combined = layers.concatenate([desc_dense, res_dense])
        
        # Final dense layers
        x = layers.Dense(128, activation='relu')(combined)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(64, activation='relu')(x)
        output = layers.Dense(self.num_labels, activation='sigmoid')(x)
        
        # Create model
        model = models.Model(
            inputs=[description_input, resolution_input],
            outputs=output
        )
        return model
    
    def create_transformer_model(self):
        """Create a dual-input Transformer model"""
        # Input layers
        description_input = layers.Input(shape=(self.max_length,), name='description_input')
        resolution_input = layers.Input(shape=(self.max_length,), name='resolution_input')
        
        # Shared embedding layer
        embedding = layers.Embedding(self.vocab_size, self.embedding_dim)
        
        # Process description path
        desc_embedded = embedding(description_input)
        desc_transformer = self._transformer_block(desc_embedded)
        desc_pooled = layers.GlobalAveragePooling1D()(desc_transformer)
        desc_dense = layers.Dense(64, activation='relu')(desc_pooled)
        
        # Process resolution path
        res_embedded = embedding(resolution_input)
        res_transformer = self._transformer_block(res_embedded)
        res_pooled = layers.GlobalAveragePooling1D()(res_transformer)
        res_dense = layers.Dense(64, activation='relu')(res_pooled)
        
        # Combine both paths
        combined = layers.concatenate([desc_dense, res_dense])
        
        # Final dense layers
        x = layers.Dense(128, activation='relu')(combined)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(64, activation='relu')(x)
        output = layers.Dense(self.num_labels, activation='sigmoid')(x)
        
        # Create model
        model = models.Model(
            inputs=[description_input, resolution_input],
            outputs=output
        )
        return model
    
    def _transformer_block(self, x, num_heads=8, ff_dim=128):
        """Create a Transformer block"""
        # Multi-head attention
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=self.embedding_dim
        )(x, x)
        
        # Add & normalize
        x1 = layers.LayerNormalization(epsilon=1e-6)(attention_output + x)
        
        # Feed-forward network
        ffn = layers.Dense(ff_dim, activation="relu")(x1)
        ffn = layers.Dense(self.embedding_dim)(ffn)
        
        # Add & normalize
        return layers.LayerNormalization(epsilon=1e-6)(ffn + x1)
    
    def create_dataset(self, descriptions, resolutions, labels, batch_size=32):
        """Create a tf.data.Dataset for training"""
        # Vectorize the inputs
        desc_data = self.description_vectorizer(descriptions)
        res_data = self.resolution_vectorizer(resolutions)
        label_data = self.label_lookup(labels)
        
        # Create dataset
        dataset = tf.data.Dataset.from_tensor_slices(
            ({'description_input': desc_data, 'resolution_input': res_data}, 
             label_data)
        )
        return dataset.shuffle(batch_size * 10).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    def train_model(self, model, train_dataset, val_dataset, epochs=10):
        """Train the model"""
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['binary_accuracy']
        )
        
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs
        )
        return history

# Example usage
def run_dual_input_classification(df):
    # Initialize classifier
    classifier = DualInputTextClassifier()
    
    # Adapt the data
    classifier.adapt_data(
        df['Description'].values,
        df['Resolution'].values,
        df['Flavors'].values
    )
    
    # Split the data
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    val_df = test_df.sample(frac=0.5)
    test_df = test_df.drop(val_df.index)
    
    # Create datasets
    train_dataset = classifier.create_dataset(
        train_df['Description'].values,
        train_df['Resolution'].values,
        train_df['Flavors'].values
    )
    
    val_dataset = classifier.create_dataset(
        val_df['Description'].values,
        val_df['Resolution'].values,
        val_df['Flavors'].values
    )
    
    # Create and train LSTM model
    lstm_model = classifier.create_lstm_model()
    print("Training LSTM model...")
    lstm_history = classifier.train_model(lstm_model, train_dataset, val_dataset)
    
    # Create and train Transformer model
    transformer_model = classifier.create_transformer_model()
    print("Training Transformer model...")
    transformer_history = classifier.train_model(transformer_model, train_dataset, val_dataset)
    
    return {
        'classifier': classifier,
        'lstm_model': lstm_model,
        'transformer_model': transformer_model,
        'lstm_history': lstm_history,
        'transformer_history': transformer_history
    }

# Example usage
if __name__ == "__main__":
    # Load your data
    df = pd.read_csv('your_data.csv')
    
    # Run the dual input classification
    results = run_dual_input_classification(df)
    
    # Make predictions with the LSTM model
    lstm_model = results['lstm_model']
    sample_desc = ["sample description"]
    sample_res = ["sample resolution"]
    
    # Prepare inputs using the vectorizer
    classifier = results['classifier']
    desc_vec = classifier.description_vectorizer(sample_desc)
    res_vec = classifier.resolution_vectorizer(sample_res)
    
    # Get predictions
    predictions = lstm_model.predict({
        'description_input': desc_vec,
        'resolution_input': res_vec
    })
