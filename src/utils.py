from src.logger import logger
from sklearn.model_selection import train_test_split


# Data splitting class 
class DataSplitter:
    def __init__(self, raw_data):
        self.raw_data = raw_data

    def split(self, test_data_size, random_state:int = None):
        try:
            # Split data into train and test
            logger.info(f"Splitting data with test size: {test_data_size}.")
            train_data, test_data = train_test_split(self.raw_data, test_size=test_data_size, random_state=random_state)
            logger.info("Data splitting successful.")
            return train_data, test_data
        except Exception as e:
            logger.error(f"Error during data splitting: {e}")
            raise