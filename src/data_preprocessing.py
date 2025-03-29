import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        
    def load_data(self, file_path):
        """
        Load data from CSV file
        """
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Successfully loaded data from {file_path}")
            return df
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    def clean_data(self, df):
        """
        Clean the dataset by handling missing values and outliers
        """
        # Handle missing values
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        # Remove outliers using IQR method
        for column in df.select_dtypes(include=[np.number]).columns:
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            df = df[~((df[column] < (Q1 - 1.5 * IQR)) | (df[column] > (Q3 + 1.5 * IQR)))]
        
        logger.info("Data cleaning completed")
        return df

    def prepare_features(self, df):
        """
        Prepare features for the model
        """
        # Convert date to datetime if present
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df['Year'] = df['Date'].dt.year
            df['Month'] = df['Date'].dt.month
            df['Day'] = df['Date'].dt.day
        
        # Select numerical features
        numerical_features = df.select_dtypes(include=[np.number]).columns
        
        # Scale features
        df[numerical_features] = self.scaler.fit_transform(df[numerical_features])
        
        logger.info("Feature preparation completed")
        return df

    def split_data(self, df, target_column, test_size=0.2):
        """
        Split data into training and testing sets
        """
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        logger.info("Data split completed")
        return X_train, X_test, y_train, y_test

def main():
    """
    Main function to demonstrate usage
    """
    preprocessor = DataPreprocessor()
    
    # Example usage
    try:
        # Load data
        df = preprocessor.load_data('data/raw/rainfall_data.csv')
        
        # Clean data
        df_clean = preprocessor.clean_data(df)
        
        # Prepare features
        df_processed = preprocessor.prepare_features(df_clean)
        
        # Split data
        X_train, X_test, y_train, y_test = preprocessor.split_data(
            df_processed, target_column='Flood_Risk'
        )
        
        # Save processed data
        X_train.to_csv('data/processed/X_train.csv', index=False)
        X_test.to_csv('data/processed/X_test.csv', index=False)
        y_train.to_csv('data/processed/y_train.csv', index=False)
        y_test.to_csv('data/processed/y_test.csv', index=False)
        
        logger.info("Data preprocessing completed successfully")
        
    except Exception as e:
        logger.error(f"Error in data preprocessing: {str(e)}")
        raise

if __name__ == "__main__":
    main() 