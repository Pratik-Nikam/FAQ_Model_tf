from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from ast import literal_eval
import pandas as pd
import numpy as np

arxiv_data = pd.read_csv(
    "https://github.com/soumik12345/multi-label-text-classification/releases/download/v0.2/arxiv_data.csv"
)
arxiv_data.head()


arxiv_data_filtered = arxiv_data.groupby("terms").filter(lambda x: len(x) > 1)
arxiv_data_filtered.shape

arxiv_data_filtered["terms"] = arxiv_data_filtered["terms"].apply(
    lambda x: literal_eval(x)
)
arxiv_data_filtered["terms"].values[:5]

test_split = 0.1

# Initial train and test split.
train_df, test_df = train_test_split(
    arxiv_data_filtered,
    test_size=test_split,
    stratify=arxiv_data_filtered["terms"].values,
)

# Splitting the test set further into validation
# and new test sets.
val_df = test_df.sample(frac=0.5)
test_df.drop(val_df.index, inplace=True)

print(f"Number of rows in training set: {len(train_df)}")
print(f"Number of rows in validation set: {len(val_df)}")
print(f"Number of rows in test set: {len(test_df)}")

terms = tf.ragged.constant(train_df["terms"].values)
lookup = tf.keras.layers.StringLookup(output_mode="multi_hot")
lookup.adapt(terms)
vocab = lookup.get_vocabulary()


def invert_multi_hot(encoded_labels):
    """Reverse a single multi-hot encoded label to a tuple of vocab terms."""
    hot_indices = np.argwhere(encoded_labels == 1.0)[..., 0]
    return np.take(vocab, hot_indices)


print("Vocabulary:\n")
print(vocab)

sample_label = train_df["terms"].iloc[0]
print(f"Original label: {sample_label}")

label_binarized = lookup([sample_label])
print(f"Label-binarized representation: {label_binarized}")

train_df["summaries"].apply(lambda x: len(x.split(" "))).describe()

max_seqlen = 150
batch_size = 128
padding_token = "<pad>"
auto = tf.data.AUTOTUNE


def make_dataset(dataframe, is_train=True):
    labels = tf.ragged.constant(dataframe["terms"].values)
    label_binarized = lookup(labels).numpy()
    dataset = tf.data.Dataset.from_tensor_slices(
        (dataframe["summaries"].values, label_binarized)
    )
    dataset = dataset.shuffle(batch_size * 10) if is_train else dataset
    return dataset.batch(batch_size)


train_dataset = make_dataset(train_df, is_train=True)
validation_dataset = make_dataset(val_df, is_train=False)
test_dataset = make_dataset(test_df, is_train=False)

text_batch, label_batch = next(iter(train_dataset))

for i, text in enumerate(text_batch[:5]):
    label = label_batch[i].numpy()[None, ...]
    print(f"Abstract: {text}")
    print(f"Label(s): {invert_multi_hot(label[0])}")
    print(" ")


vocabulary = set()
train_df["summaries"].str.lower().str.split().apply(vocabulary.update)
vocabulary_size = len(vocabulary)
print(vocabulary_size)

text_vectorizer = layers.TextVectorization(
    max_tokens=vocabulary_size, ngrams=2, output_mode="tf_idf"
)

# `TextVectorization` layer needs to be adapted as per the vocabulary from our
# training set.
with tf.device("/CPU:0"):
    text_vectorizer.adapt(train_dataset.map(lambda text, label: text))

train_dataset = train_dataset.map(
    lambda text, label: (text_vectorizer(text), label), num_parallel_calls=auto
).prefetch(auto)
validation_dataset = validation_dataset.map(
    lambda text, label: (text_vectorizer(text), label), num_parallel_calls=auto
).prefetch(auto)
test_dataset = test_dataset.map(
    lambda text, label: (text_vectorizer(text), label), num_parallel_calls=auto
).prefetch(auto)


def make_model():
    shallow_mlp_model = keras.Sequential(
        [
            layers.Dense(512, activation="relu"),
            layers.Dense(256, activation="relu"),
            layers.Dense(lookup.vocabulary_size(), activation="sigmoid"),
        ]  # More on why "sigmoid" has been used here in a moment.
    )
    return shallow_mlp_model


epochs = 1

shallow_mlp_model = make_model()
shallow_mlp_model.compile(
    loss="binary_crossentropy", optimizer="adam", metrics=["binary_accuracy"]
)

history = shallow_mlp_model.fit(
    train_dataset, validation_data=validation_dataset, epochs=epochs
)



_, binary_acc = shallow_mlp_model.evaluate(test_dataset)
print(f"Categorical accuracy on the test set: {round(binary_acc * 100, 2)}%.")

# Create a model for inference.
model_for_inference = keras.Sequential([text_vectorizer, shallow_mlp_model])

# Create a small dataset just for demoing inference.
inference_dataset = make_dataset(test_df.sample(100), is_train=False)
text_batch, label_batch = next(iter(inference_dataset))
predicted_probabilities = model_for_inference.predict(text_batch)

# Perform inference.
for i, text in enumerate(text_batch[:5]):
    label = label_batch[i].numpy()[None, ...]
    print(f"Abstract: {text}")
    print(f"Label(s): {invert_multi_hot(label[0])}")
    predicted_proba = [proba for proba in predicted_probabilities[i]]
    top_3_labels = [
        x
        for _, x in sorted(
            zip(predicted_probabilities[i], lookup.get_vocabulary()),
            key=lambda pair: pair[0],
            reverse=True,
        )
    ][:3]
    print(f"Predicted Label(s): ({', '.join([label for label in top_3_labels])})")
    print(" ")

################################################################################33

#Export
import pickle
with open("vocab.pkl", "wb") as f:
    pickle.dump(lookup.get_vocabulary(), f)



model_for_inference.export("ClassifierModel")

infer = loaded.signatures["serving_default"]
loaded = tf.saved_model.load("ClassifierModel")

input_texts = tf.constant([
    """We introduce SIRUS (Stable and Interpretable RUle Set) for regression, a\nstable rule learning algorithm which takes the form of a short and simple list\nof rules. State-of-the-art learning algorithms are often referred to as "black\nboxes" because of the high number of operations involved in their prediction\nprocess. Despite their powerful predictivity, this lack of interpretability may\nbe highly restrictive for applications with critical decisions at stake. On the\nother hand, algorithms with a simple structure-typically decision trees, rule\nalgorithms, or sparse linear models-are well known for their instability. This\nundesirable feature makes the conclusions of the data analysis unreliable and\nturns out to be a strong operational limitation. This motivates the design of\nSIRUS, which combines a simple structure with a remarkable stable behavior when\ndata is perturbed. The algorithm is based on random forests, the predictive\naccuracy of which is preserved. We demonstrate the efficiency of the method\nboth empirically (through experiments) and theoretically (with the proof of its\nasymptotic stability)."""
])
# input_texts = tf.expand_dims(input_texts, axis=-1)

output = infer(input_texts)
predictions = output["output_0"]

with open("vocab.pkl", "rb") as f:
    vocab = pickle.load(f)

top_k = 3
top_k_indices = tf.math.top_k(predictions, k=top_k).indices.numpy()

def get_predicted_labels(output_tensor, vocab, threshold=0.5, top_n=3):
    """
    Converts model output tensor to corresponding label names.

    Parameters:
    - output_tensor: Tensor containing predicted probabilities.
    - vocab: List of label names corresponding to model outputs.
    - threshold: Minimum probability to consider a label as predicted.
    - top_n: Number of top predictions to return.

    Returns:
    - List of top predicted labels.
    """
    probs = output_tensor.numpy()[0]  # Convert Tensor to NumPy array
    top_indices = probs.argsort()[-top_n:][::-1]  # Get top N indices sorted
    
    # Filter based on threshold
    predicted_labels = [vocab[i] for i in top_indices if probs[i] >= threshold]
    return predicted_labels

predicted_labels = get_predicted_labels(output["output_0"], vocab)


