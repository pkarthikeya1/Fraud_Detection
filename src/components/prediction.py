import pandas as pd
from src.components.model_building import (
    BuildTrainTestSplit,
    TrainTestSplitStrategy,
    ScalingStrategyHandler,
    MinMaxScalingStrategy,
    RandomForestModelStrategy,
    ClassifierStrategy
)
from src.db_paths import best_params

# Load the dataset
df = pd.read_parquet("path/to/your/test_data.parquet")

# Separate features and target
y = df['isFraud']  # Assuming you want to predict fraud, adjust if not
X = df.drop(columns=['isFraud'])

# Step 1: Initialize Scaling Strategy and Scale the Data
scaling_strategy = MinMaxScalingStrategy()
scaler_handler = ScalingStrategyHandler(strategy=scaling_strategy)

# Perform scaling
_, X_scaled, _ = scaler_handler.scale_data(X)

# Step 2: Initialize Data Splitting Strategy (no need to split since we are predicting)
split_strategy = TrainTestSplitStrategy()
split_handler = BuildTrainTestSplit(strategy=split_strategy)

# We do not need to split the data for prediction, so we skip this step
# X_train, X_test, y_train, y_test = split_handler.split_data(X_scaled, y, test_size=0.2, random_state=42)

# Load the trained model (assuming it has been saved or loaded)
rf_model_strategy = RandomForestModelStrategy()
classifier_strategy = ClassifierStrategy(strategy=rf_model_strategy)

# Assuming you have a trained RandomForest model saved
# (e.g., saved as 'trained_model.pkl'), you can load it here if necessary.
# For now, let's assume the model is already built and we proceed directly with predictions.

# Use the trained model to make predictions
# Make sure you have trained the model previously and have the model's parameters saved
# (i.e., load the best trained model using the best_params)

# Build the classifier (use parameters from ModelBuildingConfig.params)
model = classifier_strategy.strategy.build_classifier(X, y, **best_params)

# Make predictions on the test data
predictions = model.predict(X_scaled)

# Output predictions
print("Predictions:")
print(predictions)
