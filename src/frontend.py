from fastapi import FastAPI, File, UploadFile, HTTPException
import pandas as pd
import uvicorn
from pydantic import BaseModel
from io import BytesIO
from src.pipelines.prediction import fraud_detection

app = FastAPI()

# Define input data model
class TransactionData(BaseModel):
    amount: float
    oldbalanceOrg: float
    newbalanceOrig: float
    oldbalanceDest: float
    newbalanceDest: float
    step: int
    type: str
    nameOrig: str
    nameDest: str
    isFlaggedFraud: int

# Define accepted columns
columns = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'step', 'type', 'nameOrig', 'nameDest', 'isFlaggedFraud']

@app.post("/predict")
def predict_single(transaction: TransactionData):
    try:
        # Convert input data to DataFrame
        df = pd.DataFrame([transaction.dict()])
        
        # Call prediction function
        result, probability = fraud_detection(df)
        return {
            "prediction": result,
            "probability_non_fraud": float(probability[:, 0][0]),
            "probability_fraud": float(probability[:, 1][0])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_file")
def predict_file(file: UploadFile = File(...)):
    try:
        # Read the file
        file_extension = file.filename.split(".")[-1].lower()
        if file_extension == "csv":
            df = pd.read_csv(file.file)
        elif file_extension == "parquet":
            df = pd.read_parquet(BytesIO(file.file.read()))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type. Upload a CSV or Parquet file.")

        # Validate columns
        if not all(col in df.columns for col in columns):
            raise HTTPException(status_code=400, detail="The uploaded file must contain all required columns.")
        
        # Run predictions
        results, probabilities = [], []
        for _, row in df.iterrows():
            result, probability = fraud_detection(pd.DataFrame([row]))
            results.append(result)
            probabilities.append(probability)
        
        # Append results to DataFrame
        df['Prediction'] = results
        df['Probability_NonFraud'] = [p[:, 0][0] for p in probabilities]
        df['Probability_Fraud'] = [p[:, 1][0] for p in probabilities]
        
        return df.to_dict(orient='records')
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)