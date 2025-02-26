import pandas as pd
import numpy as np
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModel
from sklearn.preprocessing import MultiLabelBinarizer

# Step 1: Load and Prepare the Dataset
# Assuming your data is in a CSV file with columns 'description' and 'labels'
# Labels are comma-separated strings, e.g., "label1,label2,label3"
data = pd.read_csv('path/to/your/data.csv')  # Replace with your actual file path

# Convert labels from comma-separated strings to lists
data['labels'] = data['labels'].apply(lambda x: x.split(','))

# Step 2: Determine the Number of Unique Labels
mlb = MultiLabelBinarizer()
mlb.fit(data['labels'])  # Fit to find all unique labels
num_labels = len(mlb.classes_)
print(f"Number of unique labels: {num_labels}")

# Step 3: Tokenize the Text Data
model_path = 'path/to/model/folder'  # Replace with your pre-trained model path
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Tokenize the descriptions
encoded_inputs = tokenizer(
    data['description'].tolist(),
    padding=True,
    truncation=True,
    max_length=128,  # Adjust based on your text length
    return_tensors='tf'
)

# Step 4: Prepare Labels
labels = mlb.transform(data['labels'])  # Convert labels to one-hot encoded format
labels_data = tf.convert_to_tensor(labels, dtype=tf.float32)

# Step 5: Load the Base Model
base_model = TFAutoModel.from_pretrained(model_path)

# Step 6: Build the Classification Model
input_ids = tf.keras.layers.Input(shape=(128,), dtype=tf.int32, name="input_ids")
attention_mask = tf.keras.layers.Input(shape=(128,), dtype=tf.int32, name="attention_mask")

# Get the base model's output
outputs = base_model(input_ids, attention_mask=attention_mask)
pooled_output = outputs.pooler_output  # Use the pooled output for classification

# Add a classification layer
classification_layer = tf.keras.layers.Dense(num_labels, activation='sigmoid', name='classifier')(pooled_output)

# Define the full model
model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=classification_layer)

# Step 7: Compile the Model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
    loss='binary_crossentropy',  # Suitable for multi-label classification
    metrics=['accuracy']
)

# Step 8: Prepare Training Data
input_ids_data = encoded_inputs['input_ids']
attention_mask_data = encoded_inputs['attention_mask']

# Step 9: Train the Model
model.fit(
    x={'input_ids': input_ids_data, 'attention_mask': attention_mask_data},
    y=labels_data,
    epochs=3,  # Adjust epochs as needed
    batch_size=16,  # Adjust based on your hardware
    validation_split=0.2  # Use 20% of data for validation
)

# Step 10: Save the Trained Model
model.save('path/to/save/trained_model')  # Replace with your save path

# Step 11: Predict on New Data
new_texts = ["New description to classify"]  # Replace with your new text
encoded_new = tokenizer(new_texts, padding=True, truncation=True, max_length=128, return_tensors='tf')
predictions = model.predict({
    'input_ids': encoded_new['input_ids'],
    'attention_mask': encoded_new['attention_mask']
})
binary_predictions = (predictions > 0.5).astype(int)  # Threshold at 0.5
predicted_labels = mlb.inverse_transform(binary_predictions)  # Convert back to label names
print(f"Predicted labels: {predicted_labels}")
