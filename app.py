from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib

app = Flask(__name__)
CORS(app)  # Allow requests from Flutter app

# Load trained model
model = joblib.load("crop_recommendation_model.pkl")  # Ensure this file exists in the server

@app.route("/")
def home():
    return "Crop Prediction API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        
        # Extract input values
        features = [
            float(data.get("N", 0.0)),
            float(data.get("P", 0.0)),
            float(data.get("K", 0.0)),
            float(data.get("temperature", 0.0)),
            float(data.get("humidity", 0.0)),
            float(data.get("ph", 0.0)),
            float(data.get("rainfall", 0.0))
        ]

        # Convert to NumPy array and reshape
        features = np.array(features).reshape(1, -1)

        # Predict crop
        predicted_crop = model.predict(features)[0]

        return jsonify({"recommended_crop": predicted_crop})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
