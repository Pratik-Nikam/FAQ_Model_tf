import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense
from tensorflow.keras.models import Model

# Load and preprocess data
df = pd.read_csv("your_dataset.csv")
questions = df["question"].tolist()
answers = ["<start> " + ans + " <end>" for ans in df["answer"].tolist()]

# Tokenization
tokenizer = Tokenizer(filters='!"#$%&()*+,-./:;=?@[\\]^_`{|}~\t\n')
tokenizer.fit_on_texts(questions + answers)
vocab_size = len(tokenizer.word_index) + 1

# Convert text to sequences
question_sequences = tokenizer.texts_to_sequences(questions)
answer_sequences = tokenizer.texts_to_sequences(answers)

# Pad sequences
max_question_len = max(len(x) for x in question_sequences)
max_answer_len = max(len(x) for x in answer_sequences)
encoder_input_data = pad_sequences(question_sequences, maxlen=max_question_len, padding='post')
decoder_input_data = pad_sequences(answer_sequences, maxlen=max_answer_len, padding='post')

# Prepare decoder targets
decoder_target_data = np.zeros_like(decoder_input_data)
decoder_target_data[:, :-1] = decoder_input_data[:, 1:]

# Build model
encoder_inputs = Input(shape=(max_question_len,))
enc_emb = Embedding(vocab_size, 256)(encoder_inputs)
encoder_lstm = LSTM(256, return_state=True)
_, state_h, state_c = encoder_lstm(enc_emb)
encoder_states = [state_h, state_c]

decoder_inputs = Input(shape=(max_answer_len-1,))
dec_emb = Embedding(vocab_size, 256)(decoder_inputs)
decoder_lstm = LSTM(256, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=encoder_states)
decoder_dense = Dense(vocab_size, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

# Train
model.fit(
    [encoder_input_data, decoder_input_data[:, :-1]],
    decoder_target_data[:, :-1],
    batch_size=32,
    epochs=50,
    validation_split=0.2
)

# Inference setup
encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_input_h = Input(shape=(256,))
decoder_state_input_c = Input(shape=(256,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

dec_emb_inf = Embedding(vocab_size, 256)(decoder_inputs)
decoder_outputs_inf, state_h_inf, state_c_inf = decoder_lstm(dec_emb_inf, initial_state=decoder_states_inputs)
decoder_states_inf = [state_h_inf, state_c_inf]
decoder_outputs_inf = decoder_dense(decoder_outputs_inf)

decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs_inf] + decoder_states_inf
)

# Answer generation
def answer_question(input_question):
    input_seq = tokenizer.texts_to_sequences([input_question])
    input_seq = pad_sequences(input_seq, maxlen=max_question_len, padding='post')
    
    states_value = encoder_model.predict(input_seq)
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = tokenizer.word_index["<start>"]
    
    answer = []
    for _ in range(max_answer_len-1):
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = tokenizer.index_word.get(sampled_token_index, "")
        
        if sampled_word == "<end>":
            break
        answer.append(sampled_word)
        
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index
        states_value = [h, c]
    
    return " ".join(answer)

# Test
print(answer_question("What is the capital of France?"))
