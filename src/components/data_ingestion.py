import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from dataclasses import dataclass
from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split
from src.logger import logger
from src.components.data_base_connector import DatabaseConfig, SQLiteFactory


@dataclass
class DataIngestionConfig:
    raw_data_path = os.path.join("artifacts", "raw_data.parquet")
    train_data_path = os.path.join("artifacts", "train_data.parquet")
    test_data_path = os.path.join("artifacts", "test_data.parquet")


# Strategy Interface for Data Ingestion
class DataIngestionStrategy(ABC):
    @abstractmethod
    def ingest(self):
        """Ingest data from a specific source."""
        pass



# Concrete Strategy for Database Data Ingestion
class DatabaseDataIngestion(DataIngestionStrategy):
    def __init__(self, raw_data_from_db, config: DataIngestionConfig):
        self.raw_data = raw_data_from_db
        self.config = config

    def ingest(self):
        try:
            # Ingest data from the database
            logger.info("Starting data ingestion from the database.")
            self.raw_data.to_parquet(self.config.raw_data_path)
            logger.info(f"Data ingestion successful. Data saved to {self.config.raw_data_path}.")
            return self.raw_data
        except Exception as e:
            logger.error(f"Error during data ingestion from the database: {e}")
            raise


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


# Context Class that uses Strategy Pattern
class DataPipeline:
    def __init__(self, ingestion_strategy: DataIngestionStrategy, data_source):
        self.ingestion_strategy = ingestion_strategy
        self.data_source = data_source
        self.data_splitter = None

    def execute(self):
        try:
            # Ingest the data using the selected strategy
            logger.info("Starting the data pipeline execution.")
            raw_data = self.ingestion_strategy.ingest()

            # Split the data
            self.data_splitter = DataSplitter(raw_data)
            train_data, test_data = self.data_splitter.split(test_data_size=0.2, random_state=42)

            # Save the split data
            logger.info("Saving train and test data to files.")
            train_data.to_parquet(DataIngestionConfig().train_data_path)
            test_data.to_parquet(DataIngestionConfig().test_data_path)

            logger.info(f"Data pipeline execution successful. Train data saved to {DataIngestionConfig().train_data_path}, Test data saved to {DataIngestionConfig().test_data_path}.")
            return train_data, test_data

        except Exception as e:
            logger.error(f"Error during data pipeline execution: {e}")
            raise


# Usage Example
if __name__ == "__main__":
    try:
        config = DatabaseConfig()
        logger.info("Starting database operations")
        
        # Create factory
        sqlite_factory = SQLiteFactory(config)
        
        # Create connector using factory
        connector = sqlite_factory.create_connector()
        
        # Connect to database
        connector.connect()
        
        # Execute query
        results = connector.execute_query(config.sql_query)


        # Create a data pipeline using the Database Data Ingestion strategy
        db_strategy = DatabaseDataIngestion(results, DataIngestionConfig())
        pipeline = DataPipeline(ingestion_strategy=db_strategy, data_source=results)

        # Execute the pipeline
        train_data, test_data = pipeline.execute()

    except Exception as e:
        logger.error(f"Data pipeline failed with error: {e}")



