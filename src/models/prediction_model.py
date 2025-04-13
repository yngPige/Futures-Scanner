"""
Prediction Model Module for Crypto Futures Scanner

This module provides functionality to create, train, and use AI models
for predicting cryptocurrency price movements based on technical indicators.
"""

import os
import numpy as np
import joblib
from datetime import datetime
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PredictionModel:
    """Class to create and use prediction models for cryptocurrency price movements."""

    def __init__(self, model_dir='models'):
        """
        Initialize the PredictionModel.

        Args:
            model_dir (str): Directory to save/load models
        """
        self.model_dir = model_dir
        self.model = None
        self.scaler = None
        self.feature_columns = None

        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)

        logger.info("Initialized PredictionModel")

    @property
    def feature_names_in_(self):
        """
        Property to make the model compatible with scikit-learn API.
        Returns the feature columns used by the model.

        Returns:
            list: Feature column names
        """
        return self.feature_columns

    def prepare_features_target(self, df, target_column='future_return', n_forward=5):
        """
        Prepare features and target for model training.

        Args:
            df (pd.DataFrame): DataFrame with technical indicators
            target_column (str): Name of the target column
            n_forward (int): Number of periods forward for prediction

        Returns:
            tuple: X (features), y (target)
        """
        if df.empty:
            logger.warning("Empty DataFrame provided, cannot prepare features")
            return None, None

        try:
            # Create target variable (future return)
            df[target_column] = df['close'].pct_change(n_forward).shift(-n_forward)

            # Create binary target (1 for positive return, 0 for negative)
            df['target'] = np.where(df[target_column] > 0, 1, 0)

            # Drop NaN values
            df = df.dropna()

            # Select feature columns (exclude price and volume columns, and target)
            exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'target', target_column]
            exclude_cols.extend([col for col in df.columns if 'signal' in col.lower()])

            # Also exclude any columns that are all NaN
            exclude_cols.extend(df.columns[df.isna().all()].tolist())

            # Get feature columns
            feature_cols = [col for col in df.columns if col not in exclude_cols and not df[col].isna().any()]

            # Store feature columns for later use
            self.feature_columns = feature_cols

            # Extract features and target
            X = df[feature_cols]
            y = df['target']

            logger.info(f"Prepared features with {len(feature_cols)} columns and {len(X)} samples")
            return X, y

        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            return None, None

    def train_model(self, X, y, model_type='random_forest', tune_hyperparams=False):
        """
        Train a prediction model.

        Args:
            X (pd.DataFrame): Feature DataFrame
            y (pd.Series): Target Series
            model_type (str): Type of model to train ('random_forest' or 'gradient_boosting')
            tune_hyperparams (bool): Whether to tune hyperparameters using GridSearchCV

        Returns:
            bool: True if training was successful, False otherwise
        """
        if X is None or y is None:
            logger.warning("Features or target is None, cannot train model")
            return False

        try:
            # Split data into train and test sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Scale features
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

            # Select model type
            if model_type == 'random_forest':
                if tune_hyperparams:
                    # Define parameter grid
                    param_grid = {
                        'n_estimators': [50, 100, 200],
                        'max_depth': [None, 10, 20, 30],
                        'min_samples_split': [2, 5, 10],
                        'min_samples_leaf': [1, 2, 4]
                    }

                    # Create base model
                    base_model = RandomForestClassifier(random_state=42)

                    # Create GridSearchCV
                    self.model = GridSearchCV(base_model, param_grid, cv=5, scoring='f1', n_jobs=-1)
                else:
                    self.model = RandomForestClassifier(n_estimators=100, random_state=42)

            elif model_type == 'gradient_boosting':
                if tune_hyperparams:
                    # Define parameter grid
                    param_grid = {
                        'n_estimators': [50, 100, 200],
                        'learning_rate': [0.01, 0.1, 0.2],
                        'max_depth': [3, 5, 7],
                        'min_samples_split': [2, 5, 10]
                    }

                    # Create base model
                    base_model = GradientBoostingClassifier(random_state=42)

                    # Create GridSearchCV
                    self.model = GridSearchCV(base_model, param_grid, cv=5, scoring='f1', n_jobs=-1)
                else:
                    self.model = GradientBoostingClassifier(n_estimators=100, random_state=42)

            else:
                logger.error(f"Unknown model type: {model_type}")
                return False

            # Train model
            logger.info(f"Training {model_type} model...")
            self.model.fit(X_train_scaled, y_train)

            # If GridSearchCV was used, we don't need to log the details
            if tune_hyperparams:
                # Just check if the model has best_params_ attribute
                if hasattr(self.model, 'best_params_'):
                    pass  # We don't need to do anything with this

            # Evaluate model
            y_pred = self.model.predict(X_test_scaled)

            # Calculate only the metrics we need
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            # Log only the most important metrics in a concise format
            logger.info(f"Model metrics: Accuracy={accuracy:.4f}, F1={f1:.4f}")

            # Generate classification report for internal use only
            _ = classification_report(y_test, y_pred)

            # If it's a random forest, get feature importances but don't log them all
            if model_type == 'random_forest' or model_type == 'gradient_boosting':
                if tune_hyperparams:
                    model = self.model.best_estimator_
                else:
                    model = self.model

                # Get feature importances
                importances = model.feature_importances_
                indices = np.argsort(importances)[::-1]

                # Log only the top 3 most important features
                top_features = []
                for i in range(min(3, len(self.feature_columns))):
                    top_features.append(f"{self.feature_columns[indices[i]]}")

                logger.info(f"Top features: {', '.join(top_features)}")

            return True

        except Exception as e:
            logger.error(f"Error training model: {e}")
            return False

    def save_model(self, model_name=None):
        """
        Save the trained model to disk.

        Args:
            model_name (str, optional): Name to save the model as

        Returns:
            str: Path to the saved model
        """
        if self.model is None:
            logger.warning("No model to save")
            return None

        try:
            # Generate model name if not provided
            if model_name is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                model_name = f"crypto_model_{timestamp}"

            # Create full path
            model_path = os.path.join(self.model_dir, f"{model_name}.joblib")

            # Save model
            joblib.dump(self.model, model_path)

            # Save scaler
            scaler_path = os.path.join(self.model_dir, f"{model_name}_scaler.joblib")
            joblib.dump(self.scaler, scaler_path)

            # Save feature columns
            feature_path = os.path.join(self.model_dir, f"{model_name}_features.joblib")
            joblib.dump(self.feature_columns, feature_path)

            logger.info(f"Model saved to {model_path}")
            return model_path

        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return None

    def load_model(self, model_path):
        """
        Load a trained model from disk.

        Args:
            model_path (str): Path to the model file

        Returns:
            bool: True if loading was successful, False otherwise
        """
        try:
            # Load model
            self.model = joblib.load(model_path)

            # Load scaler
            scaler_path = model_path.replace('.joblib', '_scaler.joblib')
            self.scaler = joblib.load(scaler_path)

            # Load feature columns
            feature_path = model_path.replace('.joblib', '_features.joblib')
            self.feature_columns = joblib.load(feature_path)

            logger.info(f"Model loaded from {model_path}")
            return True

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False

    def predict(self, df):
        """
        Make predictions using the trained model.

        Args:
            df (pd.DataFrame): DataFrame with technical indicators

        Returns:
            pd.Series: Series with predictions
        """
        if self.model is None:
            logger.warning("No model loaded, cannot make predictions")
            return None

        if df.empty:
            logger.warning("Empty DataFrame provided, cannot make predictions")
            return None

        try:
            # Check if all required feature columns are present
            missing_cols = [col for col in self.feature_columns if col not in df.columns]
            if missing_cols:
                logger.error(f"Missing columns in input data: {missing_cols}")
                return None

            # Extract features
            X = df[self.feature_columns]

            # Scale features
            X_scaled = self.scaler.transform(X)

            # Make predictions
            predictions = self.model.predict(X_scaled)

            # Add prediction probabilities if available
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(X_scaled)
                df['prediction_probability'] = probabilities[:, 1]

            # Add predictions to DataFrame
            df['prediction'] = predictions

            logger.info(f"Made predictions for {len(df)} samples")
            return df

        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            return None


# Example usage
if __name__ == "__main__":
    # Import required modules for example
    import yfinance as yf
    from src.analysis.technical_analysis import TechnicalAnalyzer

    # Download sample data
    data = yf.download('BTC-USD', period='1y', interval='1d')

    # Add technical indicators
    analyzer = TechnicalAnalyzer()
    data_with_indicators = analyzer.add_all_indicators(data)

    # Initialize prediction model
    model = PredictionModel()

    # Prepare features and target
    X, y = model.prepare_features_target(data_with_indicators)

    # Train model
    if X is not None and y is not None:
        model.train_model(X, y, model_type='random_forest')

        # Save model
        model.save_model()

        # Make predictions
        predictions = model.predict(data_with_indicators)

        # Print results
        if predictions is not None:
            print(predictions[['close', 'prediction', 'prediction_probability']].tail())
