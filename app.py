import numpy as np
import joblib
import os
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests (Flutter compatibility)

# Install dependencies if missing
try:
    import numpy
    import sklearn
except ModuleNotFoundError:
    os.system("pip install numpy==1.24.3 joblib==1.1.0 scikit-learn==1.6.1")

# Load the trained model
MODEL_PATH = "crop_recommendation_model.pkl"

if os.path.exists(MODEL_PATH):
    try:
        model = joblib.load(MODEL_PATH)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        model = None
else:
    print("❌ Model file not found!")
    model = None

@app.route("/")
def home():
    return "🚀 Crop Prediction Model is running!"

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    try:
        data = request.get_json()

        # Extract input values with validation
        required_fields = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
        missing_fields = [field for field in required_fields if field not in data]

        if missing_fields:
            return jsonify({"error": f"Missing fields: {', '.join(missing_fields)}"}), 400
        
        # Convert inputs into a NumPy array
        features = np.array([
            data["N"], data["P"], data["K"],
            data["temperature"], data["humidity"],
            data["ph"], data["rainfall"]
        ]).reshape(1, -1)

        # Make prediction
        prediction = model.predict(features)

        return jsonify({"recommended_crop": prediction[0]})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Render dynamically assigns a port
    app.run(host="0.0.0.0", port=port)
