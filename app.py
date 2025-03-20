from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS to allow requests from Flutter

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Extracting input values
        N = data.get("N", 0)
        P = data.get("P", 0)
        K = data.get("K", 0)
        temperature = data.get("temperature", 0)
        humidity = data.get("humidity", 0)
        ph = data.get("ph", 0)
        rainfall = data.get("rainfall", 0)

        # Example prediction logic (Replace with ML model)
        if N > 50:
            recommended_crop = "Wheat"
        else:
            recommended_crop = "Rice"

        return jsonify({"recommended_crop": recommended_crop})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
