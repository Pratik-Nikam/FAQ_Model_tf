import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

# Load your dataset
# Assume 'data' is a Pandas DataFrame with 'description', 'resolution', and 'flavor' columns
# data = pd.read_csv("your_dataset.csv")
# For example purposes:
data = pd.DataFrame({
    'description': ['text1', 'text2', 'text3'],
    'resolution': ['info1', 'info2', 'info3'],
    'flavor': ['Flavor1', 'Flavor2', 'Flavor3']
})

# Combine description and resolution fields
data['text'] = data['description'] + ' ' + data['resolution']

# Encode labels (flavors)
label_encoder = LabelEncoder()
data['flavor_encoded'] = label_encoder.fit_transform(data['flavor'])
num_classes = len(label_encoder.classes_)

# Tokenize text
tokenizer = Tokenizer(num_words=10000, oov_token='<OOV>')
tokenizer.fit_on_texts(data['text'])
sequences = tokenizer.texts_to_sequences(data['text'])

# Pad sequences
max_length = 100
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')

# Prepare train-test split
X_train, X_test, y_train, y_test = train_test_split(
    padded_sequences, data['flavor_encoded'], test_size=0.2, random_state=42
)
y_train = to_categorical(y_train, num_classes=num_classes)
y_test = to_categorical(y_test, num_classes=num_classes)

# Build LSTM model
model = Sequential([
    Embedding(input_dim=10000, output_dim=128, input_length=max_length),
    LSTM(128, return_sequences=True),
    Dropout(0.2),
    LSTM(64),
    Dropout(0.2),
    Dense(128, activation='relu'),
    Dense(num_classes, activation='softmax')  # Output layer for multi-class classification
])

# Compile model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train model
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.2
)

# Evaluate model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.2f}")

# Save tokenizer and model for future use
import pickle
with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

model.save('flavor_lstm_model.h5')

def make_lstm_model():
    return keras.Sequential(
        [
            layers.Embedding(
                input_dim=vocabulary_size + 1,  # +1 for padding token
                output_dim=128,  # Embedding size
                mask_zero=True,
            ),
            layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
            layers.GlobalMaxPooling1D(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(lookup.vocabulary_size(), activation="sigmoid"),
        ]
    )

# Replace shallow MLP with LSTM-based model
lstm_model = make_lstm_model()
lstm_model.compile(
    loss="binary_crossentropy", optimizer="adam", metrics=["binary_accuracy"]
)

# Train the LSTM-based model
history = lstm_model.fit(
    train_dataset, validation_data=validation_dataset, epochs=epochs
)

# Plot the training results
plot_result("loss")
plot_result("binary_accuracy")



