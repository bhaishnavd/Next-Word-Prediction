
# Next Word Prediction using LSTM

## Project Overview

This project implements a **Next Word Prediction system** using **Natural Language Processing (NLP)** and **Deep Learning**.
The model is trained on text data and learns to predict the **most probable next word** given a sequence of words.

For example:

Input:

```
machine learning is
```

Output:

```
machine learning is a powerful tool
```

The model is built using **TensorFlow/Keras with an LSTM architecture** and is deployed with an interactive **Streamlit interface**.

---

## Technologies Used

* Python
* TensorFlow / Keras
* NumPy
* Natural Language Processing (NLP)
* Streamlit (for UI)

---

## Model Architecture

The system uses an **LSTM (Long Short-Term Memory) neural network**, which is well-suited for sequential data such as text.

Main components:

1. **Tokenizer**

   * Converts words into numerical tokens.

2. **Padding**

   * Ensures all sequences have the same length.

3. **Embedding Layer**

   * Converts tokens into dense vector representations.

4. **LSTM Layers**

   * Learns contextual relationships between words.

5. **Dense Layer (Softmax)**

   * Predicts the probability distribution of the next word.

---

## Where This System Can Be Used

Next word prediction is widely used in modern applications:

* **Smart keyboards** (Google Keyboard, SwiftKey)
* **Search engines** for query suggestions
* **Email auto-completion**
* **Chatbots and conversational AI**
* **Text generation systems**
* **Writing assistance tools**

This project demonstrates the **core concept behind these technologies**.

---

## Project Structure

```
next_word_prediction/
│
├── train_model.py        # Script to train the LSTM model
├── predict.py            # Streamlit application for prediction
├── nxtpredict.txt        # Training dataset
├── tokenizer.pkl         # Saved tokenizer
├── seq_len.pkl           # Saved sequence length
├── next_word_model.h5    # Trained model
├── requirements.txt      # Python dependencies
└── README.md
```

---

## Setup and Installation (Local System)

### 1. Clone the repository

```
git clone <your-repository-url>
cd next_word_prediction
```

### 2. Create a virtual environment (recommended)

```
python -m venv venv
```

Activate the environment:

Windows

```
venv\Scripts\activate
```

Linux / Mac

```
source venv/bin/activate
```

---

### 3. Install dependencies

```
pip install -r requirements.txt
```

---

### 4. Train the model (optional)

If you want to retrain the model:

```
python train_model.py
```

This will generate:

* `next_word_model.h5`
* `tokenizer.pkl`
* `seq_len.pkl`

---

### 5. Run the application

Start the Streamlit interface:

```
streamlit run predict.py
```

Open the browser:

```
http://localhost:8501
```

---

## Example Usage

Input:

```
deep learning
```

Output:

```
deep learning is a powerful technique
```

The system generates the **next word repeatedly to create a continuous sentence**.

---

## Future Improvements

This project currently uses an **LSTM-based architecture**.

In future versions, we plan to upgrade the model using **Transformer-based architectures**, such as:

* GPT-style language models
* Hugging Face Transformers
* Attention-based sequence modeling

These models significantly improve:

* contextual understanding
* long-range dependencies
* text generation quality

---



Developed as a **Natural Language Processing project** demonstrating the fundamentals of **language modeling and sequence prediction**.
