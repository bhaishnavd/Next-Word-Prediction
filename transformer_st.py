# import streamlit as st
# import tensorflow as tf
# # from transformers import GPT2Tokenizer, TFGPT2LMHeadModel
# from transformers import AutoTokenizer, TFAutoModelForCausalLM



# @st.cache_resource
# def load_model():
#     # tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
#     # model = TFGPT2LMHeadModel.from_pretrained("gpt2")
#     tokenizer = AutoTokenizer.from_pretrained("gpt2")
#     model = TFAutoModelForCausalLM.from_pretrained("gpt2")
#     return tokenizer, model

# tokenizer, model = load_model()

# st.title("Next Word Prediction (Transformer + TensorFlow 🚀)")

# text = st.text_input("Enter text:")

# if text:
#     inputs = tokenizer.encode(text, return_tensors="tf")
    
#     outputs = model(inputs)
#     logits = outputs.logits
    
#     next_token = tf.argmax(logits[:, -1, :], axis=-1)
#     predicted_word = tokenizer.decode(next_token.numpy()[0])
    
#     st.write("Next word:", predicted_word)

import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    return tokenizer, model


tokenizer, model = load_model()

st.title("Next Word Prediction (Transformer 🚀)")

text = st.text_input("Enter text:")

if text:
    inputs = tokenizer.encode(text, return_tensors="pt")

    with torch.no_grad():  # important
        outputs = model(inputs)
        logits = outputs.logits

    next_token = torch.argmax(logits[:, -1, :], dim=-1)
    predicted_word = tokenizer.decode(next_token.item())

    st.write("Next word:", predicted_word)