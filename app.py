import numpy as np
import joblib
import os
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load the trained model
model = joblib.load("crop_recommendation_model.pkl")

@app.route("/")
def home():
    return "🚀 Model is running!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        features = np.array(data["features"]).reshape(1, -1)
        prediction = model.predict(features)
        return jsonify({"prediction": prediction.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    # Get port from environment variable (Render assigns a dynamic port)
    port = int(os.environ.get("PORT", 10000))  # Default to 10000 if PORT is not set
    app.run(host="0.0.0.0", port=port)
