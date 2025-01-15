# Fraud Detection Model

This project implements a **Fraud Detection Model** designed to identify fraudulent financial transactions using machine learning. The model leverages a **Random Forest Classifier** to classify transactions as either fraudulent or non-fraudulent based on several transaction-related features. By utilizing data from accounts' balance changes and transaction amounts, the model efficiently detects suspicious activities.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Model Training & Hyperparameter Tuning](#model-training-and-hyperparameter-tuning)
4. [Model Performance](#model-performance)
5. [Feature Importance](#feature-importance)
6. [Usage](#usage)
7. [Installation](#installation)
8. [Files](#files)
9. [License](#license)

## Project Overview

The primary objective of this project is to detect fraudulent transactions in financial systems. The model is trained on a dataset containing several features like transaction amount, account balances before and after the transaction, and account information. By leveraging **Random Forest**, the model can differentiate between legitimate and fraudulent transactions.

### Model Approach:

- **Data Preprocessing**: The dataset undergoes preprocessing to clean and transform the data for model training. This includes handling missing values, feature scaling, and encoding categorical variables.
- **Feature Engineering**: New features are derived, such as the difference in account balances before and after the transaction, which is crucial for detecting discrepancies that indicate fraud.
- **Modeling**: A **Random Forest Classifier** is used for training the model, with hyperparameter tuning to optimize performance.
- **Evaluation**: The model is evaluated based on precision, recall, F1-score, and other classification metrics.

## Features

The dataset includes the following key features used for training the model:

- **step**: The time step of the transaction (integer).
- **type**: The type of transaction (e.g., CASH-IN, CASH-OUT, etc.).
- **amount**: The amount of the transaction.
- **nameOrig**: The name of the origin account.
- **oldbalanceOrg**: The original balance of the origin account before the transaction.
- **newbalanceOrig**: The new balance of the origin account after the transaction.
- **nameDest**: The name of the destination account.
- **oldbalanceDest**: The original balance of the destination account before the transaction.
- **newbalanceDest**: The new balance of the destination account after the transaction.

## Model Training & Hyperparameter Tuning

### Hyperparameter Tuning

To optimize the model’s performance, **grid search** was used to find the best hyperparameters for the Random Forest Classifier. The best parameters discovered are:

```python
{'n_estimators': 20, 
 'min_samples_split': 60, 
 'min_samples_leaf': 100, 
 'max_features': 'sqrt', 
 'max_depth': 4, 
 'class_weight': 'balanced', 
 'ccp_alpha': 0.0001}
```

### Best F1-Score:

The model achieved an excellent **F1-score** of `0.9713` with these hyperparameters, demonstrating its effectiveness in identifying fraudulent transactions.

## Model Performance

### Evaluation Metrics

The model was evaluated using various classification metrics, including precision, recall, and F1-score. The following performance metrics were achieved:

- **F1-Score**: `0.9894`
- **Precision**: `0.9838`
- **Recall**: `0.9951`

Here’s the **classification report** for the model:

| Class   | Precision | Recall | F1-Score | Support |
|---------|-----------|--------|----------|---------|
| 0.0 (Non-fraud) | 1.00 | 1.00 | 1.00 | 942,872 |
| 1.0 (Fraud) | 0.98 | 1.00 | 0.99 | 1,220 |
| **Accuracy** | | | **1.00** | 944,092 |
| **Macro avg** | 0.99 | 1.00 | 0.99 | 944,092 |
| **Weighted avg** | 1.00 | 1.00 | 1.00 | 944,092 |

### Cross-Validation

During **cross-validation**, the model achieved a **mean F1-score** of `0.9713` with a standard deviation of `0.0374`, indicating consistent performance across different parameter combinations.

## Feature Importance

Feature importance is an essential part of understanding the model's decision-making process. The following features were identified as the most important for detecting fraud:

| Feature             | Importance |
|---------------------|------------|
| **errorbalanceOrig** | 0.386132   |
| **newbalanceOrig**   | 0.210591   |
| **oldbalanceOrg**    | 0.162828   |
| **amount**           | 0.147096   |
| **errorbalanceDest** | 0.053315   |
| **newbalanceDest**   | 0.020096   |
| **oldbalanceDest**   | 0.019941   |

The most influential feature is **`errorbalanceOrig`**, which represents the difference between the origin account’s balance before and after the transaction. This feature, along with the transaction amount and balance changes, plays a crucial role in detecting fraudulent behavior.

## Usage

### Running the Model

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/Fraud_Detection.git
   cd Fraud_Detection
   ```

2. Install the necessary dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Train the model:
   Run the Jupyter notebook `03_EDA_and_data_preprocessing.ipynb` to perform exploratory data analysis (EDA) and preprocessing, and then proceed with training the model in the `04_model_building.ipynb`.

4. Once the model is trained, use the prediction script in `src/pipelines/prediction.py` to predict fraudulent transactions on new data.

5. The model is ready for deployment in real-time systems, and predictions can be used to flag suspicious transactions.

## Installation

To install the necessary dependencies, you can run:

```bash
pip install -r requirements.txt
```

Make sure you have Python 3.7+ and pip installed.

## Files

- **`notebooks/`**: Contains Jupyter notebooks for data exploration, preprocessing, and model building.
  - `01_data_base_connector.ipynb`: Connect to the database and load data.
  - `02_data_ingestion.ipynb`: Ingest data into the system.
  - `03_EDA_and_data_preprocessing.ipynb`: Perform exploratory data analysis and preprocess the data.
  - `04_model_building.ipynb`: Train the Random Forest model and tune hyperparameters.
  
- **`src/`**: Source code for various components like data ingestion, preprocessing, model building, and prediction.
  - `components/`: Modular components for database connection, data ingestion, preprocessing, model building, and prediction.
  - `pipelines/`: Defines prediction and model-building pipelines.
  
- **`artifacts/`**: Saved models and other output files (like trained models).

- **`requirements.txt`**: List of Python dependencies for the project.

- **`LICENSE`**: License file for the project.

- **`README.md`**: Documentation for the project.

- **`feature_importance.png`**: Feature importance plot for the trained model.

- **`learning_curve.png`**: Learning curve plot for the training process.

- **`Screenshot_UI.png`**: Sample UI visualization for model prediction results.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

### Images

![Learning Curve](./learning_curve.png)

The **learning curve** showcases the model's performance over time, allowing us to track how well the model is learning and whether there are any signs of overfitting or underfitting.

![UI Screenshot](./Screenshot_UI.png)

This is a screenshot of the UI that displays transaction predictions, showing the detection of fraudulent transactions.

```

