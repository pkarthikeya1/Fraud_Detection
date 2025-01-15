import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from abc import ABC, abstractmethod
from typing import Union, Optional, Dict, Any, Tuple, List
from dataclasses import dataclass

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, f1_score

from src.components.data_preprocessing import DataTransformationConfig
from src.db_paths import best_params
data_trans_config =DataTransformationConfig()

@dataclass
class ModelBuildingConfig:
    params =  best_params



# Step 1: Define Abstract Base Class for Data Split Strategy
class DataSplitter(ABC):
    @abstractmethod
    def split(self,
             X: Union[List, pd.DataFrame, np.ndarray, pd.Series],
             y: Optional[Union[List, pd.Series, np.ndarray]] = None,
             **kwargs) -> Dict[str, Union[np.ndarray, pd.DataFrame]]:
        """
        Abstract method to split the data into Training and Testing sets
        
        Args:
            X: Input features
            y: Target variable (optional)
            **kwargs: Additional parameters for splitting
            
        Returns:
            Dictionary containing split datasets
        """
        pass

# Step 2: Define concrete strategy for train-test split
class TrainTestSplitStrategy(DataSplitter):
    def __init__(self):
        """
        Initialize the Train-Test Split Strategy
        """
        pass
    
    def _validate_inputs(self, X, y, test_size):
        """
        Validate input parameters
        
        Args:
            X: Input features
            y: Target variable
            test_size: Proportion of test set
            
        Raises:
            ValueError: If inputs are invalid
        """
        if test_size <= 0 or test_size >= 1:
            raise ValueError("test_size must be between 0 and 1")
        
        if y is not None and len(X) != len(y):
            raise ValueError("X and y must have the same length")
    
    def _convert_to_array(self, data):
        """
        Convert input data to numpy array if needed
        
        Args:
            data: Input data
            
        Returns:
            numpy array
        """
        if isinstance(data, (pd.DataFrame, pd.Series)):
            return data.values
        elif isinstance(data, list):
            return np.array(data)
        return data
    
    def split(self,
             X: Union[List, pd.DataFrame, np.ndarray, pd.Series],
             y: Optional[Union[List, pd.Series, np.ndarray]] = None,
             test_size: float = 0.2,
             random_state: Optional[int] = None,
             shuffle: bool = True,
             stratify: Optional[Union[List, np.ndarray]] = None) -> Dict[str, Union[np.ndarray, pd.DataFrame]]:
        """
        Split the data into Training and Testing sets
        
        Args:
            X: Input features
            y: Target variable (optional)
            test_size: Proportion of test set
            random_state: Random seed for reproducibility
            shuffle: Whether to shuffle the data
            stratify: Array for stratified split
            
        Returns:
            Dictionary containing train and test splits
        """
        # Validate inputs
        self._validate_inputs(X, y, test_size)
        
        # Convert inputs to arrays if needed
        X_array = self._convert_to_array(X)
        y_array = self._convert_to_array(y) if y is not None else None
        
        # Perform split
        split_data = train_test_split(
            X_array,
            y_array,
            test_size=test_size,
            random_state=random_state,
            shuffle=shuffle,
            stratify=stratify
        )
        
        # Create return dictionary
        if y is None:
            X_train, X_test = split_data
            result = {
                'X_train': X_train,
                'X_test': X_test
            }
        else:
            X_train, X_test, y_train, y_test = split_data
            result = {
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test
            }
        
        # Add split info
        result['split_info'] = {
            'train_size': len(X_train),
            'test_size': len(X_test),
            'train_ratio': 1 - test_size,
            'test_ratio': test_size
        }
        
        return result

# Step 3: Define context class for data splitting
class BuildTrainTestSplit:
    def __init__(self, strategy: DataSplitter):
        """
        Initialize the data split handler
        
        Args:
            strategy: A concrete implementation of DataSplitter
        """
        self.strategy = strategy
    
    def set_strategy(self, strategy: DataSplitter) -> None:
        """
        Set the data splitting strategy
        
        Args:
            strategy: A concrete implementation of DataSplitter
        """
        self.strategy = strategy
    
    def split_data(self,
                  X: Union[List, pd.DataFrame, np.ndarray, pd.Series],
                  y: Optional[Union[List, pd.Series, np.ndarray]] = None,
                  **kwargs) -> Dict[str, Union[np.ndarray, pd.DataFrame]]:
        """
        Split the data using the current strategy
        
        Args:
            X: Input features
            y: Target variable (optional)
            **kwargs: Additional parameters for splitting
            
        Returns:
            Dictionary containing split datasets
        """
        return self.strategy.split(X, y, **kwargs)
    



# Step 1: Define Abstract Base Class for Scaling Strategy
class ScalingStrategy(ABC):
    @abstractmethod
    def fit(self, X: Union[List, pd.DataFrame, np.ndarray, pd.Series], 
            y: Union[List, pd.Series, np.ndarray, None] = None) -> Any:
        """
        Abstract method to fit the data using the scaler
        
        Params:
            X: Training features
            y: Target feature (optional)
            
        Returns:
            Fitted scaler instance
        """
        pass
    
    @abstractmethod
    def fit_transform(self, X: Union[List, pd.DataFrame, np.ndarray, pd.Series],
                     y: Union[List, pd.Series, np.ndarray, None] = None) -> Tuple[Any, np.ndarray, Union[np.ndarray, None]]:
        """
        Abstract method to fit and transform the data using the scaler
        
        Params:
            X: Training features
            y: Target feature (optional)
            
        Returns:
            Tuple containing (fitted scaler instance, transformed X, transformed y)
        """
        pass

# Step 2: Define Concrete Strategies for Scaling
class StandardScalingStrategy(ScalingStrategy):
    def __init__(self):
        """
        Instantiates the Standard Scaling Strategy
        """
        self.scaler = StandardScaler()
    
    def fit(self, X: Union[List, pd.DataFrame, np.ndarray, pd.Series],
            y: Union[List, pd.Series, np.ndarray, None] = None) -> StandardScaler:
        """
        Fits the data with Standard Scaler
        
        Params:
            X: Training features
            y: Target feature (optional)
            
        Returns:
            Fitted StandardScaler instance
        """
        X_array = np.array(X) if not isinstance(X, np.ndarray) else X
        self.scaler.fit(X_array.reshape(-1, 1) if X_array.ndim == 1 else X_array)
        return self.scaler
    
    def fit_transform(self, X: Union[List, pd.DataFrame, np.ndarray, pd.Series],
                     y: Union[List, pd.Series, np.ndarray, None] = None) -> Tuple[StandardScaler, np.ndarray, Union[np.ndarray, None]]:
        """
        Fits and transforms the data with Standard Scaler
        
        Params:
            X: Training features
            y: Target feature (optional)
            
        Returns:
            Tuple containing (fitted StandardScaler, transformed X, transformed y)
        """
        X_array = np.array(X) if not isinstance(X, np.ndarray) else X
        X_transformed = self.scaler.fit_transform(X_array.reshape(-1, 1) if X_array.ndim == 1 else X_array)
        
        y_transformed = None
        if y is not None:
            y_array = np.array(y) if not isinstance(y, np.ndarray) else y
            y_transformed = self.scaler.transform(y_array.reshape(-1, 1) if y_array.ndim == 1 else y_array)
        
        return self.scaler, X_transformed, y_transformed

class MinMaxScalingStrategy(ScalingStrategy):
    def __init__(self):
        """
        Instantiates the MinMax Scaling Strategy
        """
        self.scaler = MinMaxScaler()
    
    def fit(self, X: Union[List, pd.DataFrame, np.ndarray, pd.Series],
            y: Union[List, pd.Series, np.ndarray, None] = None) -> MinMaxScaler:
        """
        Fits the data with MinMax Scaler
        
        Params:
            X: Training features
            y: Target feature (optional)
            
        Returns:
            Fitted MinMaxScaler instance
        """
        X_array = np.array(X) if not isinstance(X, np.ndarray) else X
        self.scaler.fit(X_array.reshape(-1, 1) if X_array.ndim == 1 else X_array)
        return self.scaler
    
    def fit_transform(self, X: Union[List, pd.DataFrame, np.ndarray, pd.Series],
                     y: Union[List, pd.Series, np.ndarray, None] = None) -> Tuple[MinMaxScaler, np.ndarray, Union[np.ndarray, None]]:
        """
        Fits and transforms the data with MinMax Scaler
        
        Params:
            X: Training features
            y: Target feature (optional)
            
        Returns:
            Tuple containing (fitted MinMaxScaler, transformed X, transformed y)
        """
        X_array = np.array(X) if not isinstance(X, np.ndarray) else X
        X_transformed = self.scaler.fit_transform(X_array.reshape(-1, 1) if X_array.ndim == 1 else X_array)
        
        y_transformed = None
        if y is not None:
            y_array = np.array(y) if not isinstance(y, np.ndarray) else y
            y_transformed = self.scaler.transform(y_array.reshape(-1, 1) if y_array.ndim == 1 else y_array)
        
        return self.scaler, X_transformed, y_transformed

# Step 3: Define Context Class for Scalers
class ScalingStrategyHandler:
    def __init__(self, strategy: ScalingStrategy):
        """
        Initiates the scaling strategy
        
        Params:
            strategy: A concrete implementation of ScalingStrategy
        """
        self.strategy = strategy
    
    def set_strategy(self, strategy: ScalingStrategy) -> None:
        """
        Sets the scaling strategy for scaling the data
        
        Params:
            strategy: A concrete implementation of ScalingStrategy
        """
        self.strategy = strategy
    
    def scale_data(self, X: Union[List, pd.DataFrame, np.ndarray, pd.Series],
                   y: Union[List, pd.Series, np.ndarray, None] = None,
                   transform_only: bool = False) -> Union[Any, Tuple[Any, np.ndarray, Union[np.ndarray, None]]]:
        """
        Handles the scaling method
        
        Params:
            X: Training features
            y: Target feature (optional)
            transform_only: If True, only fits the scaler. If False, fits and transforms the data.
            
        Returns:
            Either the fitted scaler (if transform_only=True) or
            a tuple of (fitted scaler, transformed X, transformed y)
        """
        if transform_only:
            return self.strategy.fit(X, y)
        return self.strategy.fit_transform(X, y)
    



# Step 1: Define Abstract Base Class for Model Classifier
class ModelClassifier(ABC):
    @abstractmethod
    def build_classifier(self, 
                        X: Union[np.ndarray, pd.DataFrame], 
                        y: Union[np.ndarray, pd.Series],
                        **kwargs) -> ClassifierMixin:
        """
        Abstract method to build the classifier
        
        Args:
            X: Input training features
            y: Input labels
            **kwargs: Additional parameters for the classifier
            
        Returns:
            A fitted classifier instance
        """
        pass

    @abstractmethod
    def evaluate(self, 
                X: Union[np.ndarray, pd.DataFrame], 
                y: Union[np.ndarray, pd.Series]) -> Dict[str, Any]:
        """
        Abstract method to evaluate the classifier
        
        Args:
            X: Input test features
            y: Input test labels
            
        Returns:
            Dictionary containing evaluation metrics
        """
        pass

# Step 2: Define Concrete Product Classes
class RandomForestModelStrategy(ModelClassifier):
    def __init__(self):
        """
        Initialize the RandomForestClassifier model
        """
        self.model: Optional[RandomForestClassifier] = None
        
    def build_classifier(self, 
                        X: Union[np.ndarray, pd.DataFrame], 
                        y: Union[np.ndarray, pd.Series],
                        **kwargs) -> RandomForestClassifier:
        """
        Build and fit the Random Forest Classifier with given parameters
        
        Args:
            X: Input training features
            y: Input labels
            **kwargs: Parameters for RandomForestClassifier
            
        Returns:
            Fitted RandomForestClassifier instance
        """
        # Convert inputs to numpy arrays if needed
        X_array = np.array(X) if not isinstance(X, np.ndarray) else X
        y_array = np.array(y) if not isinstance(y, np.ndarray) else y
        
        # Create and fit the model
        self.model = RandomForestClassifier(**kwargs)
        self.model.fit(X_array, y_array)
        
        return self.model
    
    def evaluate(self, 
                X: Union[np.ndarray, pd.DataFrame], 
                y: Union[np.ndarray, pd.Series]) -> Dict[str, Any]:
        """
        Evaluate the Random Forest Classifier
        
        Args:
            X: Input test features
            y: Input test labels
            
        Returns:
            Dictionary containing evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call build_classifier first.")
        
        # Convert inputs to numpy arrays if needed
        X_array = np.array(X) if not isinstance(X, np.ndarray) else X
        y_array = np.array(y) if not isinstance(y, np.ndarray) else y
        
        # Make predictions
        y_pred = self.model.predict(X_array)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_array, y_pred),
            'classification_report': classification_report(y_array, y_pred),
            'F1-score': f1_score(y_array, y_pred),
            'feature_importance': dict(zip(range(X_array.shape[1]), 
                                        self.model.feature_importances_))
        }
        
        return metrics

# Step 3: Define Context Class
class ClassifierStrategy:
    def __init__(self, strategy: ModelClassifier):
        """
        Initialize the Classifier Strategy
        
        Args:
            strategy: A concrete implementation of ModelClassifier
        """
        self.strategy = strategy
        
    def set_strategy(self, strategy: ModelClassifier) -> None:
        """
        Set model strategy for building the model
        
        Args:
            strategy: A concrete implementation of ModelClassifier
        """
        self.strategy = strategy
        
    def _split_data(self, 
                    X: Union[np.ndarray, pd.DataFrame],
                    y: Union[np.ndarray, pd.Series],
                    test_size: float = 0.2,
                    random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Private method to split the data into training and test sets
        
        Args:
            X: Input features
            y: Input labels
            test_size: Proportion of dataset to include in the test split
            random_state: Random state for reproducibility
            
        Returns:
            Tuple containing (X_train, X_test, y_train, y_test)
        """
        return train_test_split(X, y, test_size=test_size, random_state=random_state)
        
    def build_and_evaluate(self, 
                          X: Union[np.ndarray, pd.DataFrame],
                          y: Union[np.ndarray, pd.Series],
                          split_data: bool = True,
                          test_size: float = 0.2,
                          random_state: int = 42,
                          **kwargs) -> Dict[str, Any]:
        """
        Build and evaluate the classifier
        
        Args:
            X: Input features
            y: Input labels
            split_data: Whether to split the data into train and test sets
            test_size: Proportion of dataset to include in the test split
            random_state: Random state for reproducibility
            **kwargs: Additional parameters for the classifier
            
        Returns:
            Dictionary containing model and evaluation metrics
        """
        if split_data:
            # Split the data if required
            X_train, X_test, y_train, y_test = self._split_data(
                X, y, test_size=test_size, random_state=random_state
            )
        else:
            # Use the same data for training and testing if splitting is not required
            X_train, X_test = X, X
            y_train, y_test = y, y
        
        # Build the classifier
        model = self.strategy.build_classifier(X_train, y_train, **kwargs)
        
        # Evaluate the classifier
        evaluation_metrics = self.strategy.evaluate(X_test, y_test)
        
        return {
            'model': model,
            'metrics': evaluation_metrics,
            'data': {
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test
            } if split_data else {
                'X': X,
                'y': y
            }
        }
    

if __name__ == "__main__":

    # Load the dataset
    df = pd.read_parquet(data_trans_config.transformed_train_df)

    # Separate features and target
    y = df['isFraud']
    X = df.drop(columns=['isFraud'])

    # Step 1: Initialize Scaling Strategy and Scale the Data
    scaling_strategy = MinMaxScalingStrategy()
    scaler_handler = ScalingStrategyHandler(strategy=scaling_strategy)

    _, X_scaled, _ = scaler_handler.scale_data(X)

    # Step 2: Initialize Data Splitting Strategy and Split the Data
    split_strategy = TrainTestSplitStrategy()
    split_handler = BuildTrainTestSplit(strategy=split_strategy)

    split_result = split_handler.split_data(X_scaled, y, test_size=0.2, random_state=42)
    X_train = split_result['X_train']
    X_test = split_result['X_test']
    y_train = split_result['y_train']
    y_test = split_result['y_test']

    # Step 3: Initialize Classifier Strategy and Build/Evaluate the Model
    rf_model_strategy = RandomForestModelStrategy()  # Correct class name
    classifier_strategy = ClassifierStrategy(strategy=rf_model_strategy)

    results = classifier_strategy.build_and_evaluate(
        X_train, y_train, split_data=True, **ModelBuildingConfig.params
    )

    # Print Evaluation Metrics
    print("Evaluation Metrics:")
    print(results['metrics'])
