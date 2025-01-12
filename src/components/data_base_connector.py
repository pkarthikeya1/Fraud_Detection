import os
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import sqlite3
import pandas as pd
from typing import Optional
from src.logger import logger
from src.db_paths import DataBaseName, DataBasePath, SQL_QUERY

@dataclass
class DatabaseConfig:
    db_path: str = DataBasePath
    db_name: str = DataBaseName
    sql_query: str = SQL_QUERY

# Step 1: Define the Abstract Product Interface
class Connector(ABC):
    @abstractmethod
    def connect(self) -> str:
        """Abstract method to connect to an SQL database"""
        pass
    
    @abstractmethod
    def execute_query(self, query: str) -> pd.DataFrame:
        """Abstract method to execute the query"""
        pass

# Step 2: Implement Concrete Product (SQL Connector)
class SQLite3Connector(Connector):
    def __init__(self, db_name: str):
        self.db_name = db_name
        self.connection: Optional[sqlite3.Connection] = None
        logger.info(f"Initialized SQLite3Connector with database: {db_name}")
    
    def connect(self) -> str:
        """
        Connects to a database given in db_name string
        Returns:
            str: Connection success message
        """
        try:
            self.connection = sqlite3.connect(self.db_name)
            success_msg = f"Connected to SQL database: {self.db_name}"
            logger.info(success_msg)
            return success_msg
        except sqlite3.Error as e:
            error_msg = f"Failed to connect to database {self.db_name}: {str(e)}"
            logger.error(error_msg)
            raise sqlite3.Error(error_msg)
    
    def execute_query(self, query: str) -> pd.DataFrame:
        """
        Executes a query on the connected database and returns a pandas DataFrame
        Args:
            query: str: SQL query to execute
        Returns:
            pd.DataFrame: Query results
        """
        try:
            if self.connection is None:
                error_msg = "Database not connected. Call connect() first."
                logger.error(error_msg)
                raise ConnectionError(error_msg)
            
            logger.info(f"Executing query: {query}")
            result = pd.read_sql_query(query, self.connection)
            logger.info(f"Query executed successfully. Returned {len(result)} rows")
            return result
            
        except sqlite3.Error as e:
            error_msg = f"SQL error occurred: {str(e)}"
            logger.error(error_msg)
            raise sqlite3.Error(error_msg)
        except pd.io.sql.DatabaseError as e:
            error_msg = f"Pandas database error: {str(e)}"
            logger.error(error_msg)
            raise pd.io.sql.DatabaseError(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error during query execution: {str(e)}"
            logger.error(error_msg)
            raise



# Step 3: Define Abstract Factory
class DatabaseFactory(ABC):
    @abstractmethod
    def create_connector(self) -> Connector:
        """
        Abstract method for creating a database connector
        Returns:
            Connector: A concrete connector instance
        """
        pass

# Step 4: Implement Concrete Factory
class SQLiteFactory(DatabaseFactory):
    def __init__(self, config: DatabaseConfig):
        self.config = config
        logger.info(f"Initialized SQLiteFactory with config: {config}")
    
    def create_connector(self) -> Connector:
        """
        Creates and returns a SQLite connector
        Returns:
            SQLite3Connector: A concrete SQLite connector instance
        """
        try:
            logger.info("Creating new SQLite connector")
            return SQLite3Connector(self.config.db_path)
        except Exception as e:
            error_msg = f"Failed to create SQLite connector: {str(e)}"
            logger.error(error_msg)
            raise

if __name__ == "__main__":
    try:
        # Initialize configuration
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
        logger.info(f"Successfully retrieved {len(results)} records from database")
        
    except Exception as e:
        logger.error(f"Main execution failed: {str(e)}")
        raise
    finally:
        logger.info("Database operations completed")