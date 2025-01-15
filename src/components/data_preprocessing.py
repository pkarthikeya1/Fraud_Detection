import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from abc import ABC, abstractmethod
from typing import List
from dataclasses import dataclass
import pandas as pd
import numpy as np
from src.logger import logger
from src.components.data_ingestion import DataIngestionConfig



data_ingestion_config = DataIngestionConfig()



@dataclass
class DataTransformationConfig:
    transformed_train_df = os.path.join('artifacts','transformed_train_data.parquet')
    transformed_test_df = os.path.join('artifacts','transformed_test_data.parquet')


# Step:1 -> Define Abstract Base Class for Missing Value Handling Strategy

class MissingValueHandlingStrategy(ABC):
    """
    Interface for Missinng Value Handling
    """
    @abstractmethod
    def handle(self, df:pd.DataFrame):
        """
        Abstract method to handle missing values in the DataFrame.

        Parameters:
        df (pd.DataFrame): The input DataFrame containing missing values.

        Returns:
        pd.DataFrame: The DataFrame with missing values handled.
        """
        pass



# Step:2 -> Define Concrete Strategy For Handling Missing Values

class DropMissingValuesStrategy(MissingValueHandlingStrategy):
    def __init__(self, axis=0):
        """
        Initializes the DropMissingValuesStrategy with specific parameters.

        Parameters:
        axis (int): 0 to drop rows with missing values, 1 to drop columns with missing values.
        """
        self.axis = axis

    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Drops rows or columns with missing values based on the axis.

        Parameters:
        df (pd.DataFrame): The input DataFrame containing missing values.

        Returns:
        pd.DataFrame: The DataFrame with missing values dropped.
        """
        try:
            logger.info(f"Dropping missing values with axis={self.axis}")
            df_cleaned = df.dropna(axis=self.axis)
            logger.info("Missing values dropped.")
            df_cleaned.reset_index(drop=True)
            return df_cleaned
        except Exception as e:
            logger.error(f"Error in dropping missing values: {str(e)}")
            raise


    
# Step:3 -> Define Context Class for Handling Missing Values

class MissingValueHandler:
    def __init__(self, strategy: MissingValueHandlingStrategy ):
        self.strategy = strategy
        """
        Initializes the MissingValueHandler with a specific missing value handling strategy.

        Parameters: Strategy (MissingValueHandlingStrategy): The strategy to be used for handling missing values.
        """
        pass

    def set_strategy(self, strategy:MissingValueHandlingStrategy):
        """
        Sets a strategy for handling missing values
        Parameters:
            Strategy(MissingValueHandlingStrategy) : The strategy to be used for handling missing values.
        """
        self.strategy = strategy

    def handle_missing_values(self, df:pd.DataFrame)->pd.DataFrame:
        """
        Executes the missing value handling using the current strategy.

        Parameters:
        df (pd.DataFrame): The input DataFrame containing missing values.

        Returns:
        pd.DataFrame: The DataFrame with missing values handled.
        """
        logger.info("Executing missing value handling strategy.")
        return self.strategy.handle(df)


# Step: 1 -> Define abstract base class for feature engineering
class FeatureEngineeringStrategy(ABC):

    @abstractmethod
    def feature_engineer(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Abstract method for feature engineering
        """
        pass

# Step: 2 -> Define concrete strategy for feature engineering

# Concrete strategy to drop columns
class DropColumnsStrategy(FeatureEngineeringStrategy):

    def __init__(self, columns: List):
        """
        Initiates DropColumnsStrategy
        Parameters:
            columns (List) : List of columns (features) to be dropped.
        """
        self.columns = columns

    def feature_engineer(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Drops the columns from DataFrame given the list of columns
        Parameters:
            df (pd.DataFrame): DataFrame from which columns need to be dropped
        Returns:
            df (pd.DataFrame) Returns DataFrame after dropping columns
        """
        try:
            logger.info(f"Dropping columns: {self.columns}")
            df = df.drop(columns=self.columns, axis=1)
            logger.info("Columns dropped successfully.")
            return df
        except Exception as e:
            logger.error(f"Error in dropping columns: {str(e)}")
            raise

# Concrete strategy to create new columns
class CreateColumnsStrategy(FeatureEngineeringStrategy):

    def __init__(self):
        """
        Initiates CreateColumnsStrategy
        """
        pass

    def feature_engineer(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Method to create two new columns "errorbalanceOrig", "errorbalanceDest" from the given DataFrame
        Params:
            df (pd.DataFrame): DataFrame to create new columns
        Returns:
            df (pd.DataFrame): DataFrame with two added columns
        """
        try:
            logger.info("Adding two new features: 'errorbalanceOrig' and 'errorbalanceDest'")
            df['errorbalanceOrig'] = df.newbalanceOrig + df.amount - df.oldbalanceOrg
            df['errorbalanceDest'] = df.oldbalanceDest + df.amount - df.newbalanceDest
            logger.info("New features added successfully.")
            return df
        except Exception as e:
            logger.error(f"Error in creating new features: {str(e)}")
            raise



# Concrete strategy for filling NaN in columns of object type
class FillObjectColumsWithNaN(FeatureEngineeringStrategy):

    def __init__(self, columns: List):
        """
        Initiates the FillObjectColumnsWithNaN strategy
        Params:
            columns (List): Columns of object type with empty string to be replaced with NaN.
        """
        self.columns = columns

    def feature_engineer(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Method to fill NaN in the empty rows of object type columns and convert them to float dtype
        Params:
            df (pd.DataFrame): DataFrame in which empty rows of object type columns are present.
        Returns:
            Returns DataFrame with empty rows replaced with NaN and dtype converted to float
        """
        try:
            logger.info(f"Filling NaN for columns: {self.columns}")
            for column in self.columns:
                df[column] = df[column].replace('', np.nan).astype(float)
            logger.info("NaN values filled and dtype conversion completed.")
            return df
        except Exception as e:
            logger.error(f"Error in filling NaN and converting to float dtype: {str(e)}")
            raise

# Step: 3 -> Define context class for usage of the strategy
class EngineerFeatures:
    def __init__(self, strategy: FeatureEngineeringStrategy = None):
        """
        Initiates the context class to define a specific feature engineering strategy.
        Optionally, a strategy can be set at initialization.
        
        Params:
            strategy (FeatureEngineeringStrategy): Initial strategy for feature engineering (default: None)
        """
        self.strategy = strategy
        if strategy:
            logger.info(f"Feature engineering strategy set during initialization: {strategy.__class__.__name__}")

    def set_strategy(self, strategy: FeatureEngineeringStrategy):
        """
        Sets the type of strategy required for feature engineering.
        
        Params:
            strategy (FeatureEngineeringStrategy): Strategy object type
        """
        logger.info(f"Setting feature engineering strategy: {strategy.__class__.__name__}")
        self.strategy = strategy

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Executes the feature engineering using the current strategy.
        
        Params:
            df (pd.DataFrame): DataFrame on which feature engineering will be applied
            
        Returns:
            df_feature_engineered (pd.DataFrame): DataFrame after feature engineering
        """
        if not self.strategy:
            logger.error("Feature engineering strategy is not set.")
            raise ValueError("Strategy not set. Use `set_strategy` or provide a strategy during initialization.")
        try:
            logger.info("Applying feature engineering strategy...")
            df = self.strategy.feature_engineer(df)
            logger.info("Feature engineering completed successfully.")
            return df
        except Exception as e:
            logger.error(f"Error in feature engineering process: {str(e)}")
            raise




# Usage
if __name__ == "__main__":
    # Step 1: Load the DataFrame
    df_tr = pd.read_parquet(data_ingestion_config.train_data_path)

    # Step 2: Initialize Feature Engineering Context
    feature_engineer = EngineerFeatures()

    # Apply Feature Engineering Steps
    # Drop unnecessary columns
    feature_engineer.set_strategy(DropColumnsStrategy(columns=['step', 'type', 'isFlaggedFraud', 'nameOrig', 'nameDest']))
    df_transformed = feature_engineer.engineer_features(df=df_tr)

    # Fill NaN for specific object columns and convert to float
    feature_engineer.set_strategy(FillObjectColumsWithNaN(columns=df_transformed.columns))
    df_transformed = feature_engineer.engineer_features(df=df_transformed)

    # Add new calculated columns
    feature_engineer.set_strategy(CreateColumnsStrategy())
    df_transformed = feature_engineer.engineer_features(df=df_transformed)


    # Step 3: Handle Missing Values (Drop NaNs at the end)
    missing_value_handler = MissingValueHandler(DropMissingValuesStrategy(axis=0))  # Drop rows with NaN
    df_cleaned = missing_value_handler.handle_missing_values(df=df_transformed)

    data_trans_config = DataTransformationConfig()
    df_cleaned.to_parquet(data_trans_config.transformed_train_df)
    # Final output
    print("\nFinal cleaned and feature-engineered DataFrame:\n", df_cleaned.head())
