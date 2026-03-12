import numpy as np
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.layers import Dropout

# Load data
#with open("C:\\Users\\Leelaprasad\\Desktop\\python\\nxtpredict.txt", "r", encoding="utf-8") as file:
with open(r"C:\Users\Leelaprasad\Desktop\python\nxtpredict.txt", "r", encoding="utf-8") as file:
    text = file.read().lower()

sentences = text.split("\n")

# Tokenizer
# tokenizer = Tokenizer(num_words=5000)
tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)

# total_words = len(tokenizer.word_index) + 1
total_words = min(5000, len(tokenizer.word_index) + 1)
# Create sequences
input_sequences = []

for sentence in sentences:
    token_list = tokenizer.texts_to_sequences([sentence])[0]

    for i in range(1, len(token_list)):
        input_sequences.append(token_list[:i+1])

max_sequence_len = max(len(x) for x in input_sequences)

input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_len, padding="pre")

X = input_sequences[:, :-1]
y = input_sequences[:, -1]

#y = tf.keras.utils.to_categorical(y, num_classes=total_words)
model = Sequential([
    Embedding(input_dim=5000, output_dim=128),
    LSTM(150, return_sequences=True),
    Dropout(0.2),
    LSTM(100),
    Dense(5000, activation="softmax")
])

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

model.fit(X, y, epochs=30,batch_size=64)
# Model
# model = Sequential([
#     Embedding(total_words, 64, input_length=max_sequence_len-1),
#     LSTM(100),
#     Dense(total_words, activation="softmax")
# ])

# #model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
# model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
# # Train
# model.fit(X, y, epochs=10)

# Save model
model.save("next_word_model.h5")

# Save tokenizer
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

# Save sequence length
with open("seq_len.pkl", "wb") as f:
    pickle.dump(max_sequence_len, f)

print("Model saved successfully!")