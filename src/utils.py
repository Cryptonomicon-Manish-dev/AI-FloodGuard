import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_directories():
    """
    Create necessary directories for the project
    """
    directories = [
        'data/raw',
        'data/processed',
        'models',
        'visualizations',
        'notebooks'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Created directory: {directory}")

def validate_data(df):
    """
    Validate input data format and content
    """
    required_columns = ['Date', 'Location', 'Rainfall_Amount']
    
    # Check for required columns
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Check for missing values
    missing_values = df[required_columns].isnull().sum()
    if missing_values.any():
        logger.warning(f"Found missing values:\n{missing_values}")
    
    # Check for invalid values
    if (df['Rainfall_Amount'] < 0).any():
        raise ValueError("Found negative rainfall values")
    
    # Check date format
    try:
        pd.to_datetime(df['Date'])
    except Exception as e:
        raise ValueError(f"Invalid date format: {str(e)}")
    
    logger.info("Data validation completed successfully")

def calculate_rolling_features(df, window_sizes=[3, 7, 14, 30]):
    """
    Calculate rolling features for rainfall data
    """
    features = pd.DataFrame()
    
    for window in window_sizes:
        # Rolling mean
        features[f'rainfall_mean_{window}d'] = df['Rainfall_Amount'].rolling(
            window=window, min_periods=1
        ).mean()
        
        # Rolling sum
        features[f'rainfall_sum_{window}d'] = df['Rainfall_Amount'].rolling(
            window=window, min_periods=1
        ).sum()
        
        # Rolling max
        features[f'rainfall_max_{window}d'] = df['Rainfall_Amount'].rolling(
            window=window, min_periods=1
        ).max()
        
        # Rolling std
        features[f'rainfall_std_{window}d'] = df['Rainfall_Amount'].rolling(
            window=window, min_periods=1
        ).std()
    
    return features

def prepare_weather_features(df):
    """
    Prepare weather-related features
    """
    # Convert date to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Extract time-based features
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['Season'] = df['Month'].apply(lambda x: (x % 12 + 3) // 3)
    
    # Calculate monsoon season indicator
    df['IsMonsoon'] = ((df['Month'] >= 6) & (df['Month'] <= 9)).astype(int)
    
    return df

def calculate_risk_level(probability):
    """
    Convert probability to risk level
    """
    if probability < 0.3:
        return 'Low'
    elif probability < 0.7:
        return 'Medium'
    else:
        return 'High'

def save_results(df, filename):
    """
    Save results to CSV file
    """
    try:
        df.to_csv(filename, index=False)
        logger.info(f"Results saved to {filename}")
    except Exception as e:
        logger.error(f"Error saving results: {str(e)}")
        raise

def load_results(filename):
    """
    Load results from CSV file
    """
    try:
        df = pd.read_csv(filename)
        logger.info(f"Results loaded from {filename}")
        return df
    except Exception as e:
        logger.error(f"Error loading results: {str(e)}")
        raise

def main():
    """
    Main function to demonstrate usage
    """
    # Setup directories
    setup_directories()
    
    # Example usage
    try:
        # Load sample data
        df = pd.DataFrame({
            'Date': pd.date_range(start='2023-01-01', periods=100),
            'Location': ['Chandigarh'] * 100,
            'Rainfall_Amount': np.random.normal(10, 5, 100)
        })
        
        # Validate data
        validate_data(df)
        
        # Calculate features
        rolling_features = calculate_rolling_features(df)
        weather_features = prepare_weather_features(df)
        
        # Combine features
        final_df = pd.concat([df, rolling_features, weather_features], axis=1)
        
        # Save results
        save_results(final_df, 'data/processed/features.csv')
        
        logger.info("Example processing completed successfully")
        
    except Exception as e:
        logger.error(f"Error in example processing: {str(e)}")
        raise

if __name__ == "__main__":
    main() 