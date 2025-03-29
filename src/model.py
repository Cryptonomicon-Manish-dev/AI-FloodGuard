import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import cross_val_score
import joblib
import logging
import argparse

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FloodRiskPredictor:
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        
    def load_data(self, X_train_path, y_train_path, X_test_path, y_test_path):
        """
        Load training and testing data
        """
        try:
            X_train = pd.read_csv(X_train_path)
            y_train = pd.read_csv(y_train_path)
            X_test = pd.read_csv(X_test_path)
            y_test = pd.read_csv(y_test_path)
            
            logger.info("Data loaded successfully")
            return X_train, y_train, X_test, y_test
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    def train(self, X_train, y_train):
        """
        Train the model
        """
        try:
            self.model.fit(X_train, y_train.values.ravel())
            logger.info("Model training completed")
            
            # Perform cross-validation
            cv_scores = cross_val_score(self.model, X_train, y_train.values.ravel(), cv=5)
            logger.info(f"Cross-validation scores: {cv_scores}")
            logger.info(f"Average CV score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
            
        except Exception as e:
            logger.error(f"Error in model training: {str(e)}")
            raise

    def evaluate(self, X_test, y_test):
        """
        Evaluate the model performance
        """
        try:
            y_pred = self.model.predict(X_test)
            y_pred_proba = self.model.predict_proba(X_test)[:, 1]
            
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted'),
                'recall': recall_score(y_test, y_pred, average='weighted'),
                'f1': f1_score(y_test, y_pred, average='weighted'),
                'roc_auc': roc_auc_score(y_test, y_pred_proba)
            }
            
            logger.info("Model evaluation metrics:")
            for metric, value in metrics.items():
                logger.info(f"{metric}: {value:.3f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error in model evaluation: {str(e)}")
            raise

    def predict(self, X):
        """
        Make predictions on new data
        """
        try:
            predictions = self.model.predict(X)
            probabilities = self.model.predict_proba(X)
            return predictions, probabilities
        except Exception as e:
            logger.error(f"Error in prediction: {str(e)}")
            raise

    def save_model(self, model_path):
        """
        Save the trained model
        """
        try:
            joblib.dump(self.model, model_path)
            logger.info(f"Model saved to {model_path}")
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise

    def load_model(self, model_path):
        """
        Load a trained model
        """
        try:
            self.model = joblib.load(model_path)
            logger.info(f"Model loaded from {model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

def main():
    parser = argparse.ArgumentParser(description='Flood Risk Prediction Model')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--predict', action='store_true', help='Make predictions')
    parser.add_argument('--input', type=str, help='Input data path for predictions')
    parser.add_argument('--model_path', type=str, default='models/flood_risk_model.joblib',
                      help='Path to save/load the model')
    
    args = parser.parse_args()
    
    predictor = FloodRiskPredictor()
    
    if args.train:
        # Load training data
        X_train, y_train, X_test, y_test = predictor.load_data(
            'data/processed/X_train.csv',
            'data/processed/y_train.csv',
            'data/processed/X_test.csv',
            'data/processed/y_test.csv'
        )
        
        # Train model
        predictor.train(X_train, y_train)
        
        # Evaluate model
        predictor.evaluate(X_test, y_test)
        
        # Save model
        predictor.save_model(args.model_path)
        
    elif args.predict:
        if not args.input:
            logger.error("Please provide input data path for predictions")
            return
            
        # Load model
        predictor.load_model(args.model_path)
        
        # Load input data
        X = pd.read_csv(args.input)
        
        # Make predictions
        predictions, probabilities = predictor.predict(X)
        
        # Save predictions
        results = pd.DataFrame({
            'prediction': predictions,
            'probability': probabilities[:, 1]
        })
        results.to_csv('data/processed/predictions.csv', index=False)
        logger.info("Predictions saved to data/processed/predictions.csv")

if __name__ == "__main__":
    main() 