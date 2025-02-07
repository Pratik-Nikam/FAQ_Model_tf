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



"""
