from flask import Flask, render_template, request, jsonify
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import re
import string

app = Flask(__name__)

# Load model and tokenizer from Hugging Face
model_name = "sandbox338/hatespeech"  
try:
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f"Successfully loaded model: {model_name}")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    raise e

# Class labels mapping
class_labels = {
    0: "Non-hate speech",
    1: "Political hate speech",
    2: "offensive language"
}

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\busername_\w+\b', '', text)
    text = re.sub(r'#\w+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return re.sub(r'\s+', ' ', text).strip()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    try:
        text = request.form.get('text')
        if not text:
            return jsonify({"error": "No text provided"}), 400
        
        cleaned_text = clean_text(text)
        inputs = tokenizer(cleaned_text, return_tensors="pt", padding=True, truncation=True, max_length=128)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1).tolist()[0]
        predicted_class = torch.argmax(logits, dim=1).item()
        
        response = {
            "text": text,
            "cleaned_text": cleaned_text,
            "predicted_class": class_labels[predicted_class],
            "class_id": predicted_class,
            "probabilities": {
                class_labels[0]: probabilities[0],
                class_labels[1]: probabilities[1],
                class_labels[2]: probabilities[2]
            }
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
