import os
import sys
import pandas as pd
import streamlit as st

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

# Import the backend function
from src.pipelines.prediction import fraud_detection

# Define the columns
columns = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'step', 'type', 'nameOrig', 'nameDest', 'isFlaggedFraud']


def predict_datapoint():
    st.title('Fraudulent Transaction Detection')

    # Columns for layout
    col = st.columns(2)

    # Input fields matching the expected attributes
    amount = col[1].number_input('Amount', step=1, min_value=0, max_value=15000000)
    oldbalanceOrg = col[0].number_input('Old Balance (Origin)', step=1, min_value=0, max_value=15000000)
    newbalanceOrig = col[0].number_input('New Balance (Origin)', step=1, min_value=0, max_value=15000000)
    oldbalanceDest = col[1].number_input('Old Balance (Destination)', step=1, min_value=0, max_value=15000000)
    newbalanceDest = col[0].number_input('New Balance (Destination)', step=1, min_value=0, max_value=15000000)
    step = col[1].number_input('Step', step=1, min_value=1, max_value=745)
    type = col[0].selectbox('Type', ['Transfer', 'Payment', 'CashIn', 'CashOut'])
    nameOrig = col[1].text_input('Origin Account Name')
    nameDest = col[0].text_input('Destination Account Name')
    isFlaggedFraud = col[1].number_input('Is Flagged Fraud', min_value=0, max_value=1, step=1)

    # Prepare DataFrame for prediction
    data = {
        'amount': [amount],
        'oldbalanceOrg': [oldbalanceOrg],
        'newbalanceOrig': [newbalanceOrig],
        'oldbalanceDest': [oldbalanceDest],
        'newbalanceDest': [newbalanceDest],
        'step': [step],
        'type': [type],
        'nameOrig': [nameOrig],
        'nameDest': [nameDest],
        'isFlaggedFraud': [isFlaggedFraud],
    }
    df_pred = pd.DataFrame(data)

    return df_pred


def predict_from_file(uploaded_file, file_type):
    try:
        # Load the file based on its type
        if file_type == "csv":
            df = pd.read_csv(uploaded_file)
        elif file_type == "parquet":
            df = pd.read_parquet(uploaded_file)

        # Ensure all required columns are present
        if not all(col in df.columns for col in columns):
            st.error("The uploaded file must contain all required columns.")
            return

        # Call the backend function to predict
        results = []
        probabilities = []

        for _, row in df.iterrows():
            result, probability = fraud_detection(pd.DataFrame([row]))
            results.append(result)
            probabilities.append(probability)

        # Add predictions and probabilities to the DataFrame
        df['Prediction'] = results
        df['Probability_NonFraud'] = [p[:, 0][0] for p in probabilities]
        df['Probability_Fraud'] = [p[:, 1][0] for p in probabilities]

        st.success("Predictions complete!")
        st.dataframe(df)

        # Provide a download link for the results
        if file_type == "csv":
            csv_result = df.to_csv(index=False)
            st.download_button("Download Predictions (CSV)", data=csv_result, file_name="predictions.csv", mime="text/csv")
        elif file_type == "parquet":
            parquet_path = "predictions.parquet"
            df.to_parquet(parquet_path, index=False)
            with open(parquet_path, "rb") as f:
                st.download_button("Download Predictions (Parquet)", data=f, file_name="predictions.parquet", mime="application/octet-stream")

    except Exception as e:
        st.error(f"An error occurred while processing the file: {str(e)}")


# Main function
def main():
    st.sidebar.header("Input Method")
    input_method = st.sidebar.radio("Select input method", ("Manual Entry", "Upload File"))

    if input_method == "Manual Entry":
        df_pred = predict_datapoint()

        if st.button('Predict'):
            try:
                # Call the backend function to predict
                result, probability = fraud_detection(df=df_pred)
                st.success(f"{result}, probability of non-fraudulent is {probability[:, 0][0]:.2%}, probability of fraudulent is {probability[:, 1][0]:.2%}")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

    elif input_method == "Upload File":
        uploaded_file = st.file_uploader("Upload a CSV or Parquet file", type=["csv", "parquet"])
        if uploaded_file:
            file_extension = uploaded_file.name.split(".")[-1].lower()
            if file_extension in ["csv", "parquet"]:
                predict_from_file(uploaded_file, file_type=file_extension)
            else:
                st.error("Unsupported file type. Please upload a CSV or Parquet file.")


# Run the app
if __name__ == "__main__":
    main()
