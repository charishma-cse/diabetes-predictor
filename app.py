from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
import joblib
import os
import traceback

app = Flask(__name__)
CORS(app, supports_credentials=True)

MODEL_DIR = "models"

models = {}

model_files = {
    "logistic": "logistic.pkl",
    "decision_tree": "decision_tree.pkl",
    "random_forest": "random_forest.pkl",
    "knn": "knn.pkl"
}

for name, filename in model_files.items():
    path = os.path.join(MODEL_DIR, filename)
    if os.path.exists(path):
        models[name] = joblib.load(path)
        print(f"Loaded {name} model")
    else:
        print(f"{filename} not found")

scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
if os.path.exists(scaler_path):
    scaler = joblib.load(scaler_path)
    print("Scaler loaded")
else:
    raise FileNotFoundError("scaler.pkl not found inside models folder")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        required_fields = ["glucose", "bmi", "age", "pregnancies", "model"]
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing input: {field}"}), 400

        glucose = float(data["glucose"])
        bmi = float(data["bmi"])
        age = float(data["age"])
        pregnancies = float(data["pregnancies"])
        model_name = data["model"]

        if model_name not in models:
            return jsonify({
                "error": f"Model '{model_name}' is not available."
            }), 400

        features = np.array([[pregnancies, glucose, bmi, age]])
        features_scaled = scaler.transform(features)

        model = models[model_name]
        prediction = model.predict(features_scaled)[0]
        result = "Diabetic" if prediction == 1 else "Not Diabetic"

        return jsonify({
            "result": result,
            "model": model_name
        })

    except Exception as e:
        print("PREDICTION ERROR:", str(e))
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")
    

if __name__ == "__main__":
    print("Diabetes Prediction Server Started")
    print("http://127.0.0.1:5000")
    print("Available models:", list(models.keys()))
    app.run(debug=True)
