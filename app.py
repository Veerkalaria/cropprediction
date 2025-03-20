import numpy as np
import joblib
import os
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests from Flutter

# Load the trained model
model = joblib.load("crop_recommendation_model.pkl")

@app.route("/")
def home():
    return "🚀 Crop Prediction Model is running!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        
        # Extract individual values from the request
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
    port = int(os.environ.get("PORT", 10000))  # Default to 10000 if PORT is not set
    app.run(host="0.0.0.0", port=port)
