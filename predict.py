import numpy as np
import tensorflow as tf
import pickle
import streamlit as st
from tensorflow.keras.preprocessing.sequence import pad_sequences


model = tf.keras.models.load_model("next_word_model.h5")


with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)


with open("seq_len.pkl", "rb") as f:
    max_sequence_len = pickle.load(f)

def generate_text(seed_text, next_words=10):
    seed_text = seed_text.lower()
    for _ in range (next_words):
       

        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding="pre")

        predicted = model.predict(token_list, verbose=0)
        predicted_index = np.argmax(predicted)
        
        output_word = tokenizer.index_word.get(predicted_index)

        
        seed_text = seed_text + " " + output_word
    return seed_text
#   return None


# while True:
#     text = input("\nEnter text (or exit): ")

#     if text == "exit":
#         break

#     word = predict_next_word(text)
#     print("Predicted word:", word)
st.title("Next Word Prediction")

text= st.text_input("enter sentence")
if st.button("Generate Sentence"):
    result = generate_text(text, 20)
    st.write(result)