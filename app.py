from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib
import os

app = Flask(__name__)
CORS(app)

# Load trained model with error handling
model_path = "crop_recommendation_model.pkl"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file {model_path} not found")
model = joblib.load(model_path)

@app.route("/")
def home():
    return "Crop Prediction API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        features = [
            float(data.get("N", 0.0)),
            float(data.get("P", 0.0)),
            float(data.get("K", 0.0)),
            float(data.get("temperature", 0.0)),
            float(data.get("humidity", 0.0)),
            float(data.get("ph", 0.0)),
            float(data.get("rainfall", 0.0))
        ]
        features = np.array(features).reshape(1, -1)
        predicted_crop = model.predict(features)[0]
        return jsonify({"recommended_crop": predicted_crop})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
