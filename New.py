import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

# 1. Create and adapt preprocessing layers first
# --------------------------------------------------
# Sample data (replace with your dataset)
train_texts = [
    "Quantum physics research",
    "Neural network applications",
    "Deep learning advancements",
    "Computer vision breakthroughs",
    "Natural language processing"
]
train_labels = np.array([
    [1, 0, 1],
    [1, 1, 0],
    [0, 1, 1],
    [1, 0, 1],
    [0, 1, 0]
], dtype="float32")

# Create and adapt text vectorization layer
text_vectorizer = layers.TextVectorization(
    max_tokens=1000,
    output_mode='count',
    standardize='lower_and_strip_punctuation',
    name='text_vectorizer'
)
text_vectorizer.adapt(tf.data.Dataset.from_tensor_slices(train_texts).batch(128))

# 2. Build model with functional API
# --------------------------------------------------
def build_model():
    inputs = layers.Input(shape=(1,), dtype=tf.string, name='text_input')
    x = text_vectorizer(inputs)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dense(32, activation='relu')(x)
    outputs = layers.Dense(train_labels.shape[1], activation='sigmoid', name='predictions')(x)
    
    return models.Model(inputs, outputs)

model = build_model()

# 3. Compile and train
# --------------------------------------------------
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['binary_accuracy']
)

# Create dataset
train_dataset = tf.data.Dataset.from_tensor_slices((train_texts, train_labels))
train_dataset = train_dataset.batch(2).prefetch(tf.data.AUTOTUNE)

# Train
history = model.fit(train_dataset, epochs=3)

# 4. Verify model input/output
# --------------------------------------------------
test_text = ["Test physics and learning"]
print("Raw text prediction:", model.predict(test_text))

# 5. Save with explicit layer registration
# --------------------------------------------------
# Save the model
model.save("text_classifier.keras", 
          save_format="keras",
          save_traces=True)

# 6. Load with custom objects
# --------------------------------------------------
# Reload model with custom objects
loaded_model = tf.keras.models.load_model(
    "text_classifier.keras",
    custom_objects={'TextVectorization': layers.TextVectorization}
)

# Test loaded model
print("Loaded model prediction:", loaded_model.predict(test_text))


######


import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

# 1. Create preprocessing layers with explicit serialization
# ----------------------------------------------------------
class TextVectorizerWrapper(layers.Layer):
    """Custom wrapper to ensure proper serialization"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.vectorizer = layers.TextVectorization(
            max_tokens=10000,
            output_mode='multi_hot',
            standardize='lower_and_strip_punctuation',
            name='text_vectorizer'
        )
        
    def adapt(self, data):
        self.vectorizer.adapt(data)
        
    def call(self, inputs):
        return self.vectorizer(inputs)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "max_tokens": self.vectorizer.max_tokens,
            "output_mode": self.vectorizer.output_mode,
            "standardize": self.vectorizer.standardize,
            "name": self.vectorizer.name
        })
        return config

# 2. Create and adapt preprocessing
# ----------------------------------------------------------
# Sample data
train_texts = [
    "Machine learning research", 
    "Deep neural networks",
    "Computer vision applications",
    "Natural language processing",
    "AI in healthcare"
]
train_labels = np.array([
    [1, 0, 1],
    [1, 1, 0],
    [0, 1, 1],
    [1, 0, 1],
    [0, 1, 0]
], dtype="float32")

# Create and adapt vectorizer
text_vectorizer = TextVectorizerWrapper()
text_vectorizer.adapt(tf.data.Dataset.from_tensor_slices(train_texts).batch(32))

# 3. Build model with functional API
# ----------------------------------------------------------
def create_model():
    inputs = layers.Input(shape=(1,), dtype=tf.string, name='text_input')
    x = text_vectorizer(inputs)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu')(x)
    outputs = layers.Dense(train_labels.shape[1], activation='sigmoid')(x)
    
    return models.Model(inputs, outputs)

model = create_model()

# 4. Compile and train
# ----------------------------------------------------------
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['binary_accuracy']
)

# Create dataset
dataset = tf.data.Dataset.from_tensor_slices((train_texts, train_labels))
dataset = dataset.batch(2).prefetch(tf.data.AUTOTUNE)

# Train
history = model.fit(dataset, epochs=5)

# 5. Verify preprocessing
# ----------------------------------------------------------
test_phrase = "New AI developments in machine learning"
print("Vectorized shape:", text_vectorizer(test_phrase).shape)
print("Sample prediction:", model.predict([test_phrase]))

# 6. Save with custom objects
# ----------------------------------------------------------
model.save("text_model.keras", save_format="keras")

# 7. Load with proper custom object handling
# ----------------------------------------------------------
loaded_model = models.load_model(
    "text_model.keras",
    custom_objects={
        'TextVectorizerWrapper': TextVectorizerWrapper,
        'TextVectorization': layers.TextVectorization
    }
)

# Test loaded model
print("Loaded model prediction:", loaded_model.predict([test_phrase]))
