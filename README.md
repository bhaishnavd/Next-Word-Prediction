#  Next Word Prediction (LSTM ➝ Transformer)

## 📌 Project Overview

This project demonstrates **Next Word Prediction** using two approaches:

*  **LSTM-based Deep Learning Model (Initial Version)**
*  **Transformer-based Model using GPT-2 (Upgraded Version)**

It predicts the **next word** given an input sequence and showcases the transition from traditional deep learning to modern NLP architectures.

---

##  Project Evolution

### 🔹 Initial Version (LSTM)

* Built using **TensorFlow / Keras**
* Used **Embedding + LSTM layers**
* Trained on dataset for sequence prediction

### 🔹 Updated Version (Transformer 🚀)

* Upgraded to **GPT-2 (Transformer)**
* Uses **Hugging Face Transformers**
* Implemented using **PyTorch**
* No training required (pre-trained model)

---

## ⚙️ Tech Stack

### LSTM Version

* Python
* TensorFlow / Keras
* NumPy

### Transformer Version

* Python
* Hugging Face Transformers
* PyTorch
* Streamlit

---

## 🏗️ Model Details

### 📌 LSTM

* Sequential Model
* Embedding Layer
* LSTM Layer
* Dense Layer

### 📌 Transformer (GPT-2)

* Pre-trained GPT-2 model
* AutoTokenizer
* AutoModelForCausalLM

---

## 💻 How It Works

### LSTM Approach

1. Text preprocessing
2. Sequence creation
3. Model training
4. Predict next word

### Transformer Approach

1. Input text is tokenized
2. GPT-2 generates next token probabilities
3. Highest probability token is selected
4. Token is decoded into word

---

## 📂 Project Structure

```bash
README.md                     # Project documentation
next_word_model_transformer.py  # Transformer model logic (updated version)
transformer_st.py              # Streamlit web app
train_model.py                 # LSTM model training
predict.py                     # LSTM prediction script
```

---

## ▶️ Run the Project

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

---

### 2. Run Transformer Web App (Recommended 🚀)

```bash
python -m streamlit run transformer_st.py
```

---

## 📸 Example

**Input:**

```
I love artificial
```

**Output:**

```
Next word: intelligence
```

---

## ✨ Features

* LSTM-based prediction (custom trained)
* Transformer-based prediction (GPT-2)
* Interactive UI with Streamlit
* Real-time next word prediction

---

## 🚀 Why Transformer Upgrade?

* Better context understanding
* Handles long-range dependencies
* More accurate predictions
* Uses state-of-the-art NLP model

---

## 📝 Commit Highlights

* ✅ Initial commit → LSTM model implementation
* ✅ Added training and prediction scripts
* 🚀 Updated project → Migrated from LSTM to Transformer (GPT-2)

---

## 🔮 Future Improvements

* Top-K predictions
* Sentence completion
* Fine-tuning GPT-2
* Deployment on cloud

---

## 🙌 Acknowledgements

* Hugging Face Transformers
* GPT-2 Model
* TensorFlow / Keras
* Streamlit

---

⭐ This project highlights the transition from traditional RNNs (LSTM) to modern Transformer-based NLP models.
