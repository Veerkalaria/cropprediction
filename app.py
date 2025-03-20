import os
import numpy as np
import joblib
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
    port = int(os.environ.get("PORT", 5000))  # Get PORT from env or default to 5000
    app.run(host="0.0.0.0", port=port)
