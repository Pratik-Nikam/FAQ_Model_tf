import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd

class DescriptionClassifierWithResolution:
    def __init__(self, vocab_size=10000, max_length=128, embedding_dim=128):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.embedding_dim = embedding_dim
        
        # Text vectorization layers
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
        """Create a model that uses both inputs during training but only description during inference"""
        # Input layers
        description_input = layers.Input(shape=(self.max_length,), name='description_input')
        resolution_input = layers.Input(shape=(self.max_length,), name='resolution_input')
        
        # Shared embedding layer
        embedding = layers.Embedding(self.vocab_size, self.embedding_dim)
        
        # Process description path
        desc_embedded = embedding(description_input)
        desc_lstm = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(desc_embedded)
        desc_lstm = layers.Bidirectional(layers.LSTM(32))(desc_lstm)
        desc_features = layers.Dense(64, activation='relu')(desc_lstm)
        
        # Process resolution path (only used during training)
        res_embedded = embedding(resolution_input)
        res_lstm = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(res_embedded)
        res_lstm = layers.Bidirectional(layers.LSTM(32))(res_lstm)
        res_features = layers.Dense(64, activation='relu')(res_lstm)
        
        # Combine features with attention mechanism
        attention_weights = layers.Dense(1, activation='sigmoid')(res_features)
        enhanced_desc_features = layers.multiply([desc_features, attention_weights])
        
        # Final classification layers
        x = layers.Dense(128, activation='relu')(enhanced_desc_features)
        x = layers.Dropout(0.5)(x)
        output = layers.Dense(self.num_labels, activation='sigmoid')(x)
        
        # Create training model (uses both inputs)
        training_model = models.Model(
            inputs=[description_input, resolution_input],
            outputs=output,
            name='training_model'
        )
        
        # Create inference model (uses only description)
        inference_model = models.Model(
            inputs=description_input,
            outputs=training_model.layers[-1](
                training_model.layers[-2](
                    training_model.layers[-3](desc_features)
                )
            ),
            name='inference_model'
        )
        
        return training_model, inference_model
    
    def create_transformer_model(self):
        """Create a Transformer-based model with the same training/inference pattern"""
        # Input layers
        description_input = layers.Input(shape=(self.max_length,), name='description_input')
        resolution_input = layers.Input(shape=(self.max_length,), name='resolution_input')
        
        # Shared embedding layer
        embedding = layers.Embedding(self.vocab_size, self.embedding_dim)
        
        # Process description path
        desc_embedded = embedding(description_input)
        desc_transformer = self._transformer_block(desc_embedded)
        desc_pooled = layers.GlobalAveragePooling1D()(desc_transformer)
        desc_features = layers.Dense(64, activation='relu')(desc_pooled)
        
        # Process resolution path
        res_embedded = embedding(resolution_input)
        res_transformer = self._transformer_block(res_embedded)
        res_pooled = layers.GlobalAveragePooling1D()(res_transformer)
        res_features = layers.Dense(64, activation='relu')(res_pooled)
        
        # Combine with attention
        attention_weights = layers.Dense(1, activation='sigmoid')(res_features)
        enhanced_desc_features = layers.multiply([desc_features, attention_weights])
        
        # Final layers
        x = layers.Dense(128, activation='relu')(enhanced_desc_features)
        x = layers.Dropout(0.5)(x)
        output = layers.Dense(self.num_labels, activation='sigmoid')(x)
        
        # Create both models
        training_model = models.Model(
            inputs=[description_input, resolution_input],
            outputs=output,
            name='training_model'
        )
        
        inference_model = models.Model(
            inputs=description_input,
            outputs=training_model.layers[-1](
                training_model.layers[-2](
                    training_model.layers[-3](desc_features)
                )
            ),
            name='inference_model'
        )
        
        return training_model, inference_model
    
    def _transformer_block(self, x, num_heads=8, ff_dim=128):
        """Helper function to create a Transformer block"""
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=self.embedding_dim
        )(x, x)
        x1 = layers.LayerNormalization(epsilon=1e-6)(attention_output + x)
        ffn = layers.Dense(ff_dim, activation="relu")(x1)
        ffn = layers.Dense(self.embedding_dim)(ffn)
        return layers.LayerNormalization(epsilon=1e-6)(ffn + x1)
    
    def create_training_dataset(self, descriptions, resolutions, labels, batch_size=32):
        """Create a training dataset with both inputs"""
        desc_data = self.description_vectorizer(descriptions)
        res_data = self.resolution_vectorizer(resolutions)
        label_data = self.label_lookup(labels)
        
        dataset = tf.data.Dataset.from_tensor_slices(
            ({'description_input': desc_data, 'resolution_input': res_data}, 
             label_data)
        )
        return dataset.shuffle(batch_size * 10).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    def create_inference_dataset(self, descriptions, batch_size=32):
        """Create a dataset for inference (description only)"""
        desc_data = self.description_vectorizer(descriptions)
        return tf.data.Dataset.from_tensor_slices(desc_data).batch(batch_size).prefetch(tf.data.AUTOTUNE)

# Example usage
def run_classification(df):
    # Initialize classifier
    classifier = DescriptionClassifierWithResolution()
    
    # Adapt the data
    classifier.adapt_data(
        df['Description'].values,
        df['Resolution'].values,
        df['Flavors'].values
    )
    
    # Split data
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    val_df = test_df.sample(frac=0.5)
    test_df = test_df.drop(val_df.index)
    
    # Create training datasets
    train_dataset = classifier.create_training_dataset(
        train_df['Description'].values,
        train_df['Resolution'].values,
        train_df['Flavors'].values
    )
    
    val_dataset = classifier.create_training_dataset(
        val_df['Description'].values,
        val_df['Resolution'].values,
        val_df['Flavors'].values
    )
    
    # Create and train LSTM models
    train_lstm, inference_lstm = classifier.create_lstm_model()
    
    train_lstm.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['binary_accuracy']
    )
    
    print("Training LSTM model...")
    lstm_history = train_lstm.fit(train_dataset, validation_data=val_dataset, epochs=10)
    
    return {
        'classifier': classifier,
        'training_model': train_lstm,
        'inference_model': inference_lstm,
        'history': lstm_history
    }

# Example of how to use for inference
if __name__ == "__main__":
    # Load your data
    df = pd.read_csv('your_data.csv')
    
    # Train the model
    results = run_classification(df)
    
    # Make predictions using only description
    inference_model = results['inference_model']
    classifier = results['classifier']
    
    # Example prediction
    sample_description = ["sample description text"]
    prediction = inference_model.predict(
        classifier.description_vectorizer(sample_description)
    )
