from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "*"}})

# Load Model and Tokenizer
MODEL_PATH = r"C:\Users\dassu\OneDrive\Documents\C folder\Python programs\app______\saved_bert_model"

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    print(model)
    model.eval()
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit(1)  # Exit if model loading fails

level_1_mapping = {0: "Non-Misogynistic", 1: "Misogynistic"}
level_2_mapping = {0: "Pejorative", 1: "Treatment", 2: "Derogation", 3: "Personal Attack", 4: "Non-Misogynistic Attack", 5: "Counter Speech", 6: "None"}
#level_3_mapping = {0: "Non-Misogynistic Personal Attack", 1: "Counter Speech", 2: "None of the Categories"}

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Prediction function
def classify_text(text):
    inputs = tokenizer(text, truncation=True, padding="max_length", max_length=512, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask, output_attentions=True)
        logits_1, logits_2 = outputs
        attentions = outputs.attentions  # Extracts attention weights

    # Get predicted label
    prediction_1 = torch.argmax(logits_1, dim=1).item()
    prediction_2 = torch.argmax(logits_2, dim=1).item()
    #prediction_3 = torch.argmax(logits_3, dim=1).item()
    
    label_1 = level_1_mapping.get(prediction_1, "Unknown")
    label_2 = level_2_mapping.get(prediction_2, "Unknown")
    #label_3 = level_3_mapping.get(prediction_3, "Unknown")

    highlighted_words = []
    if hasattr(model, "attentions"):
    # Extract attention scores
        attention_scores = attentions[-1].mean(dim=1).squeeze(0)  # Last layer's average attention
        tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze().tolist())
        scores = attention_scores.tolist()

    # ✅ FIX: Properly zip token attention pairs
        token_attention = list(zip(tokens, scores))

    # Sort words by highest attention
        token_attention.sort(key=lambda x: x[1], reverse=True)
        highlighted_words = [word for word, score in token_attention[:3] if word not in ["[CLS]", "[SEP]", "[PAD]"]]

    return {"Level 1": label_1,
            "Level 2": label_2,
            '''"Level 3": label_3,'''
            "Highlighted Words": highlighted_words}

# API Route
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        text = data.get("text", "").strip()

        if not text:
            return jsonify({"error": "No text provided"}), 400

        result = classify_text(text)
        return jsonify(result)  # ✅ FIX: Return result directly

    except Exception as e:
        print(f"❌ Error in prediction: {e}")
        return jsonify({"error": "Internal Server Error", "details": str(e)}), 500

# Run the Flask app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
