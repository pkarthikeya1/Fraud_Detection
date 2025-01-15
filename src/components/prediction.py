# backend.py
import joblib
import pandas as pd
from src.components.model_building import ScalingStrategyHandler, MinMaxScalingStrategy, RandomForestModelStrategy, ClassifierStrategy
from src.components.data_preprocessing import (
    DropColumnsStrategy, CreateColumnsStrategy, FillObjectColumsWithNaN, MissingValueHandler, DropMissingValuesStrategy, EngineerFeatures
)

# Load necessary configurations
scaling_strategy = MinMaxScalingStrategy()
scaler_handler = ScalingStrategyHandler(strategy=scaling_strategy)

# Define numerical and discrete columns
numerical_columns = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
discrete_columns = ['step', 'type']

# Initialize the classifier
rf_model_strategy = RandomForestModelStrategy()
classifier_strategy = ClassifierStrategy(strategy=rf_model_strategy)

# Function for fraud detection
def fraud_detection(df):
    # Create a DataFrame from user inputs
    input_data = df

    # Apply Feature Engineering Steps
    feature_engineer = EngineerFeatures()

    feature_engineer.set_strategy(DropColumnsStrategy(columns=['step', 'type', 'isFlaggedFraud', 'nameOrig', 'nameDest']))
    input_data_transformed = feature_engineer.engineer_features(df=input_data)

    # Step 1: Handle Missing Values
    missing_value_handler = MissingValueHandler(DropMissingValuesStrategy(axis=0))  # Drop rows with NaN
    input_data_cleaned = missing_value_handler.handle_missing_values(input_data_transformed)

    # Step 2: Feature Engineering
    # 2.1 Drop unnecessary columns (if any)
  

    # 2.2 Fill NaN for specific object columns and convert to float
    feature_engineer.set_strategy(FillObjectColumsWithNaN(columns=input_data_cleaned.columns))
    input_data_cleaned = feature_engineer.engineer_features(df=input_data_cleaned)

    # 2.3 Add new calculated columns
    feature_engineer.set_strategy(CreateColumnsStrategy())
    input_data_transformed = feature_engineer.engineer_features(df=input_data_cleaned)

    # Step 3: Scaling
    _, X_scaled, _ = scaler_handler.scale_data(input_data_transformed)

    # Load trained model and make prediction
    model = joblib.load("artifacts/model.pkl")  # Assuming your model is already trained and saved in the strategy
    prediction = model.predict(X_scaled)
    probability = model.predict_proba(X_scaled)
    # Return the prediction result
    result = ('Fraudulent' if prediction[0] == 1 else 'Non-Fraudulent')
    return  result, probability
