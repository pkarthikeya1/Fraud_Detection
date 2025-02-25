import os
import sys
from flask import Flask, request, jsonify
import pandas as pd
from io import BytesIO
from src.pipelines.prediction import fraud_detection

app = Flask(__name__)

# Define accepted columns (for validation)
columns = [
    'amount', 'oldbalanceOrg', 'newbalanceOrig', 
    'oldbalanceDest', 'newbalanceDest', 'step', 
    'type', 'nameOrig', 'nameDest', 'isFlaggedFraud'
]

@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        df = pd.DataFrame([data])
        result, probability = fraud_detection(df)
        output = {
            "prediction": result,
            "probability_non_fraud": float(probability[:, 0][0]),
            "probability_fraud": float(probability[:, 1][0])
        }
        return jsonify(output)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/api/predict_file", methods=["POST"])
def predict_file():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    file_extension = file.filename.split(".")[-1].lower()

    try:
        if file_extension == "csv":
            df = pd.read_csv(file)
        elif file_extension == "parquet":
            df = pd.read_parquet(BytesIO(file.read()))
        else:
            return jsonify({"error": "Unsupported file format. Upload CSV or Parquet."}), 400

        # Validate required columns
        if not all(col in df.columns for col in columns):
            return jsonify({"error": "The uploaded file must contain all required columns."}), 400

        results = []
        probabilities = []

        for _, row in df.iterrows():
            row_df = pd.DataFrame([row])
            result, probability = fraud_detection(row_df)
            results.append(result)
            probabilities.append(probability)

        df["Prediction"] = results
        df["Probability_NonFraud"] = [p[:, 0][0] for p in probabilities]
        df["Probability_Fraud"] = [p[:, 1][0] for p in probabilities]

        return df.to_json(orient="records")

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8080)
