# import tensorflow as tf
# from transformers import GPT2Tokenizer,TFAutoModelForCausalLM

# class TFNextWordPredictor:
#     def __init__(self):
#         print("Loading TensorFlow Transformer model...")
        
#         self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
#         self.model = TFAutoModelForCausalLM.from_pretrained("gpt2")
    
#     def predict_next_word(self, text):
#         inputs = self.tokenizer.encode(text, return_tensors="tf")
        
#         outputs = self.model(inputs)
#         logits = outputs.logits
        
#         next_token_id = tf.argmax(logits[:, -1, :], axis=-1)
#         predicted_word = self.tokenizer.decode(next_token_id.numpy()[0])
        
#         return predicted_word

#     def generate_text(self, text, max_length=30):
#         inputs = self.tokenizer.encode(text, return_tensors="tf")
        
#         output = self.model.generate(
#             inputs,
#             max_length=max_length,
#             num_return_sequences=1,
#             no_repeat_ngram_size=2
#         )
        
#         return self.tokenizer.decode(output[0], skip_special_tokens=True)


# # Run
# if __name__ == "__main__":
#     model = TFNextWordPredictor()
    
#     text = "I love artificial intelligence"
    
#     print("Input:", text)
#     print("Next word:", model.predict_next_word(text))
#     print("Generated text:", model.generate_text(text))

print("RUNNING UPDATED FILE ✅")

import torch
from transformers import GPT2Tokenizer,AutoModelForCausalLM

class TFNextWordPredictor:
    def __init__(self):
        print("Loading TensorFlow Transformer model...")
        
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.model = AutoModelForCausalLM.from_pretrained("gpt2")
        
        
        # 🔥 FIX: GPT-2 padding issue
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def predict_next_word(self, text):
        inputs = self.tokenizer.encode(text, return_tensors="pt")
        
        outputs = self.model(inputs)
        logits = outputs.logits
        
        next_token_id = torch.argmax(logits[:, -1, :], axis=-1)
        predicted_word = self.tokenizer.decode(next_token_id.item())
        
        return predicted_word

    def generate_text(self, text, max_length=30):
        inputs = self.tokenizer.encode(text, return_tensors="pt")
        
        output = self.model.generate(
            inputs,
            max_length=max_length,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            pad_token_id=self.tokenizer.eos_token_id   # 🔥 important
        )
        
        return self.tokenizer.decode(output[0], skip_special_tokens=True)


# Run
if __name__ == "__main__":
    model = TFNextWordPredictor()
    
    text = "I love artificial intelligence"
    
    print("Input:", text)
    print("Next word:", model.predict_next_word(text))
    print("Generated text:", model.generate_text(text))