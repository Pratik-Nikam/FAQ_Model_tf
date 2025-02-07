"""
## Dataset preview
"""

text_batch, label_batch = next(iter(train_dataset))

for i, text in enumerate(text_batch[:5]):
    label = label_batch[i].numpy()[None, ...]
    print(f"Abstract: {text}")
    print(f"Label(s): {invert_multi_hot(label[0])}")
    print(" ")

"""
## Vectorization
"""

# Create TextVectorization layer
text_vectorizer = layers.TextVectorization(
    max_tokens=vocabulary_size,
    ngrams=2,
    output_mode="tf_idf",
    name="text_vectorizer"
)

# Adapt vectorizer to training data
with tf.device("/CPU:0"):
    text_vectorizer.adapt(train_dataset.map(lambda text, label: text))

"""
## Create a text classification model with integrated preprocessing
"""

def make_model():
    inputs = keras.Input(shape=(1,), dtype=tf.string, name="text_input")
    x = text_vectorizer(inputs)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dense(256, activation="relu")(x)
    outputs = layers.Dense(lookup.vocabulary_size(), activation="sigmoid")(x)
    
    model = keras.Model(inputs, outputs)
    return model

"""
## Train the model
"""

epochs = 20

# Create datasets WITHOUT map operations (raw text input)
raw_train_dataset = train_dataset.prefetch(auto)
raw_validation_dataset = validation_dataset.prefetch(auto)
raw_test_dataset = test_dataset.prefetch(auto)

shallow_mlp_model = make_model()
shallow_mlp_model.compile(
    loss="binary_crossentropy",
    optimizer="adam",
    metrics=["binary_accuracy"]
)

history = shallow_mlp_model.fit(
    raw_train_dataset,
    validation_data=raw_validation_dataset,
    epochs=epochs
)

"""
## Save and reload the model
"""

# Save the entire model (including text vectorization)
shallow_mlp_model.save("saved_model.keras")

# Load with custom objects (if needed)
loaded_model = tf.keras.models.load_model(
    "saved_model.keras",
    custom_objects={'TextVectorization': layers.TextVectorization}
)

"""
## Verify end-to-end prediction
"""

sample_text = ["Machine learning in quantum physics research"]
print("Prediction:", loaded_model.predict(sample_text))
