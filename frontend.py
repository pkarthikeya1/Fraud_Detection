import os
import sys
# Add project root to Python path (adjust the relative path as needed)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from flask import Flask, render_template, request, jsonify, redirect, url_for
import pandas as pd
from io import BytesIO
from src.pipelines.prediction import fraud_detection

app = Flask(__name__)

# Define accepted columns (used for file upload validation)
columns = [
    'amount', 'oldbalanceOrg', 'newbalanceOrig', 
    'oldbalanceDest', 'newbalanceDest', 'step', 
    'type', 'nameOrig', 'nameDest', 'isFlaggedFraud'
]

@app.route("/")
def home():
    return render_template("index.html")

# Route for direct prediction via form (GET shows form, POST processes prediction)
@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        try:
            # Retrieve and cast form values
            data = {
                "amount": float(request.form["amount"]),
                "oldbalanceOrg": float(request.form["oldbalanceOrg"]),
                "newbalanceOrig": float(request.form["newbalanceOrig"]),
                "oldbalanceDest": float(request.form["oldbalanceDest"]),
                "newbalanceDest": float(request.form["newbalanceDest"]),
                "step": int(request.form["step"]),
                "type": request.form["type"],
                "nameOrig": request.form["nameOrig"],
                "nameDest": request.form["nameDest"],
                "isFlaggedFraud": int(request.form["isFlaggedFraud"])
            }
            df = pd.DataFrame([data])
            result, probability = fraud_detection(df)
            output = {
                "prediction": result,
                "probability_non_fraud": float(probability[:, 0][0]),
                "probability_fraud": float(probability[:, 1][0])
            }
            return render_template("predict.html", result=output, data=data)
        except Exception as e:
            error = str(e)
            return render_template("predict.html", error=error)
    else:
        return render_template("predict.html")

# Route for file upload prediction (GET shows file form, POST processes the file)
@app.route("/predict_file", methods=["GET", "POST"])
def predict_file():
    if request.method == "POST":
        if "file" not in request.files:
            error = "No file part"
            return render_template("predict_file.html", error=error)
        file = request.files["file"]
        if file.filename == "":
            error = "No file selected"
            return render_template("predict_file.html", error=error)
        file_extension = file.filename.split(".")[-1].lower()
        try:
            if file_extension == "csv":
                df = pd.read_csv(file)
            elif file_extension == "parquet":
                df = pd.read_parquet(BytesIO(file.read()))
            else:
                error = "Unsupported file type. Please upload a CSV or Parquet file."
                return render_template("predict_file.html", error=error)
            
            # Validate that required columns are present
            if not all(col in df.columns for col in columns):
                error = "The uploaded file must contain all required columns."
                return render_template("predict_file.html", error=error)
            
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
            
            # Convert the dataframe to an HTML table to display results
            table_html = df.to_html(classes="table table-striped", index=False)
            return render_template("predict_file.html", table=table_html)
        except Exception as e:
            error = str(e)
            return render_template("predict_file.html", error=error)
    else:
        return render_template("predict_file.html")

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8080)
