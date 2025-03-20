import numpy as np
import joblib
import os
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

# Load the trained model
try:
    model = joblib.load("crop_recommendation_model.pkl")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@app.route("/")
def home():
    return "🚀 Crop Prediction Model is running!"

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded properly"}), 500

    try:
        data = request.get_json()

        # Extract individual values with default fallbacks
        N = data.get("N", 0)
        P = data.get("P", 0)
        K = data.get("K", 0)
        temperature = data.get("temperature", 0)
        humidity = data.get("humidity", 0)
        ph = data.get("ph", 0)
        rainfall = data.get("rainfall", 0)

        # Create features array in the format expected by the model
        features = np.array([N, P, K, temperature, humidity, ph, rainfall]).reshape(1, -1)

        # Make prediction
        prediction = model.predict(features)

        # Return the result
        return jsonify({"recommended_crop": prediction[0]})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    # Get port from environment variable (Render assigns a dynamic port)
    port = int(os.environ.get("PORT", 5000))  # Default to 5000 if PORT is not set
    app.run(host="0.0.0.0", port=port)
