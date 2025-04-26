from flask import Flask, render_template, request, jsonify
from transformers import AutoModelForSequenceClassification, AutoTokenizer, PreTrainedModel
import torch
import re
import string
import os
from huggingface_hub import hf_hub_download

app = Flask(__name__)

# Define model repo and parameters
model_name = "sandbox338/swahili-hatespeech"
model_directory = "swahili_hate_speech_model"  

# Download the model files to a local directory
try:
    # Create a temporary directory to store the model
    os.makedirs("downloaded_model", exist_ok=True)
    
    # Download specific model files
    model_file = hf_hub_download(
        repo_id=model_name,
        filename=f"{model_directory}/model.safetensors",  
        repo_type="model",
        local_dir="downloaded_model"
    )
    
    # Download tokenizer files 
    tokenizer_files = ["config.json", "tokenizer_config.json", "vocab.txt", "special_tokens_map.json"]
    for file in tokenizer_files:
        try:
            hf_hub_download(
                repo_id=model_name,
                filename=f"{model_directory}/{file}",
                repo_type="model",
                local_dir="downloaded_model"
            )
        except Exception:
            print(f"Could not find {file}, continuing...")
    
    # Load model and tokenizer from the downloaded files
    model = torch.load("downloaded_model/pytorch_model.bin")
    tokenizer = AutoTokenizer.from_pretrained("downloaded_model")
    
except Exception as e:
    print(f"Error loading model: {str(e)}")
    print("Attempting alternative loading method...")
    # Try direct loading if structure allows
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        subfolder=model_directory,
        ignore_mismatched_sizes=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        subfolder=model_directory
    )

# Class labels mapping
class_labels = {
    0: "Non-hate speech",
    1: "Political hate speech",
    2: "Offensive language"
}

# Rest of your code remains the same
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\busername_\w+\b', '', text)
    text = re.sub(r'#\w+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    try:
        # Get text from form
        text = request.form.get('text')
        
        if not text:
            return jsonify({"error": "No text provided"}), 400
        
        # Clean and tokenize the text
        cleaned_text = clean_text(text)
        inputs = tokenizer(cleaned_text, return_tensors="pt", padding=True, truncation=True, max_length=128)
        
        # Get predictions
        with torch.no_grad():
            # Adjust based on your model's interface
            if isinstance(model, PreTrainedModel):
                outputs = model(**inputs)
                logits = outputs.logits
            else:
                # For custom models
                outputs = model(inputs)
                logits = outputs
        
        # Process predictions
        probabilities = torch.softmax(logits, dim=1).tolist()[0]
        predicted_class = torch.argmax(logits, dim=1).item()
        
        # Prepare response
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
