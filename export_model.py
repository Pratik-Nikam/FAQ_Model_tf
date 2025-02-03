from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from ast import literal_eval
import pandas as pd
import numpy as np

arxiv_data = pd.read_csv("arxiv_data.csv")
arxiv_data.head()

arxiv_data = arxiv_data[~arxiv_data["titles"].duplicated()]

# Filtering the rare terms.
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

mlb = MultiLabelBinarizer()
mlb.fit_transform(train_df["terms"])
mlb.classes_

sample_label = train_df["terms"].iloc[0]
print(f"Original label: {sample_label}")

label_binarized = mlb.transform([sample_label])
print(f"Label-binarized representation: {label_binarized}")

max_seqlen = 150
batch_size = 128
padding_token = "<pad>"


def unify_text_length(text, label):
    # Split the given abstract and calculate its length.
    word_splits = tf.strings.split(text, sep=" ")
    sequence_length = tf.shape(word_splits)[0]
    
    # Calculate the padding amount.
    padding_amount = max_seqlen - sequence_length
    
    # Check if we need to pad or truncate.
    if padding_amount > 0:
        unified_text = tf.pad([text], [[0, padding_amount]], constant_values="<pad>")
        unified_text = tf.strings.reduce_join(unified_text, separator="")
    else:
        unified_text = tf.strings.reduce_join(word_splits[:max_seqlen], separator=" ")
    
    # The expansion is needed for subsequent vectorization.
    return tf.expand_dims(unified_text, -1), label


def make_dataset(dataframe, is_train=True):
    label_binarized = mlb.transform(dataframe["terms"].values)
    dataset = tf.data.Dataset.from_tensor_slices(
        (dataframe["summaries"].values, label_binarized)
    )
    dataset = dataset.shuffle(batch_size * 10) if is_train else dataset
    dataset = dataset.map(unify_text_length).cache()
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
	
	
train_dataset = make_dataset(train_df, is_train=True)
validation_dataset = make_dataset(val_df, is_train=False)
test_dataset = make_dataset(test_df, is_train=False)

train_df["total_words"] = train_df["summaries"].str.split().str.len()
vocabulary_size = train_df["total_words"].max()
print(f"Vocabulary size: {vocabulary_size}")

text_vectorizer = layers.TextVectorization(
    max_tokens=vocabulary_size, ngrams=2, output_mode="tf_idf"
)

# `TextVectorization` needs to be adapted as per the vocabulary from our
# training set.
with tf.device("/CPU:0"):
    text_vectorizer.adapt(train_dataset.map(lambda text, label: text))


def make_model():
    shallow_mlp_model = keras.Sequential(
        [
            text_vectorizer,
            layers.Dense(512, activation="relu"),
            layers.Dense(256, activation="relu"),
            layers.Dense(len(mlb.classes_), activation="sigmoid"),
        ]
    )
    return shallow_mlp_model
	
shallow_mlp_model = make_model()
shallow_mlp_model.summary()

epochs = 1

shallow_mlp_model.compile(
    loss="binary_crossentropy", optimizer="adam", metrics=["categorical_accuracy"]
)

history = shallow_mlp_model.fit(train_dataset, validation_data=validation_dataset, epochs=epochs)

_, categorical_acc = shallow_mlp_model.evaluate(test_dataset)
print(f"Categorical accuracy on the test set: {round(categorical_acc * 100, 2)}%.")

text_batch, label_batch = next(iter(test_dataset))
predicted_probabilities = shallow_mlp_model.predict(text_batch)

for i, text in enumerate(text_batch[:5]):
    label = label_batch[i].numpy()[None, ...]
    print(f"Abstract: {text[0]}")
    print(f"Label(s): {mlb.inverse_transform(label)[0]}")
    predicted_proba = [proba for proba in predicted_probabilities[i]]
    top_3_labels = [x for _, x in sorted(zip(predicted_probabilities[i], mlb.classes_), key=lambda pair: pair[0], reverse=True)][:3]
    print(f"Predicted Label(s): ({', '.join([label for label in top_3_labels])})")
    print(" ")
	

shallow_mlp_model.export("my_saved_model")

loaded = tf.saved_model.load("my_saved_model")

infer = loaded.signatures["serving_default"]

input_texts = tf.constant([
    """We introduce SIRUS (Stable and Interpretable RUle Set) for regression, a\nstable rule learning algorithm which takes the form of a short and simple list\nof rules. State-of-the-art learning algorithms are often referred to as "black\nboxes" because of the high number of operations involved in their prediction\nprocess. Despite their powerful predictivity, this lack of interpretability may\nbe highly restrictive for applications with critical decisions at stake. On the\nother hand, algorithms with a simple structure-typically decision trees, rule\nalgorithms, or sparse linear models-are well known for their instability. This\nundesirable feature makes the conclusions of the data analysis unreliable and\nturns out to be a strong operational limitation. This motivates the design of\nSIRUS, which combines a simple structure with a remarkable stable behavior when\ndata is perturbed. The algorithm is based on random forests, the predictive\naccuracy of which is preserved. We demonstrate the efficiency of the method\nboth empirically (through experiments) and theoretically (with the proof of its\nasymptotic stability)."""
])
input_texts = tf.expand_dims(input_texts, axis=-1)


output = infer(input_texts)

predictions = output["output_0"]

top_k = 3
top_k_indices = tf.math.top_k(predictions, k=top_k).indices.numpy()

actual_labels = [mlb.classes_[i] for i in top_k_indices]



https://github.com/kmkarakaya/Deep-Learning-Tutorials/blob/master/Text_Vectorization_Use_Save_Upload.ipynb
