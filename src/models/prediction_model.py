"""
Prediction Model Module for Crypto Futures Scanner

This module provides functionality to create, train, and use AI models
for predicting cryptocurrency price movements based on technical indicators.
"""

import os
import numpy as np
import pandas as pd
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
        if df is None:
            logger.error("None DataFrame provided, cannot prepare features")
            return None, None

        if df.empty:
            logger.error("Empty DataFrame provided, cannot prepare features")
            return None, None

        # Check if we have enough data points for prediction
        if len(df) <= n_forward:
            logger.error(f"Not enough data points for {n_forward}-period forward prediction. Got {len(df)} rows.")
            return None, None

        # Log the shape of the DataFrame for debugging
        logger.info(f"Initial DataFrame shape: {df.shape}")
        logger.info(f"DataFrame columns: {df.columns.tolist()}")

        try:
            # Create target variable (future return)
            df[target_column] = df['close'].pct_change(n_forward).shift(-n_forward)

            # Create binary target (1 for positive return, 0 for negative)
            df['target'] = np.where(df[target_column] > 0, 1, 0)

            # Drop NaN values
            df_clean = df.dropna()

            # Check if we still have data after dropping NaN values
            if df_clean.empty:
                logger.error("All rows contain NaN values after creating target variable")
                # Count NaN values in each column for debugging
                nan_counts = df.isna().sum()
                logger.info(f"NaN counts in columns: {nan_counts.to_dict()}")
                return None, None

            # Select feature columns (exclude price and volume columns, and target)
            exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'target', target_column]
            exclude_cols.extend([col for col in df_clean.columns if 'signal' in col.lower()])

            # Also exclude any columns that are all NaN
            exclude_cols.extend(df_clean.columns[df_clean.isna().all()].tolist())

            # Get feature columns
            feature_cols = [col for col in df_clean.columns if col not in exclude_cols and not df_clean[col].isna().any()]

            # Check if we have any feature columns
            if not feature_cols:
                logger.error("No valid feature columns found after filtering")
                # Log excluded columns for debugging
                logger.info(f"Excluded columns: {exclude_cols}")
                logger.info(f"Remaining columns with NaN: {[col for col in df_clean.columns if col not in exclude_cols and df_clean[col].isna().any()]}")
                return None, None

            # Store feature columns for later use
            self.feature_columns = feature_cols

            # Extract features and target
            X = df_clean[feature_cols]
            y = df_clean['target']

            # Final check to ensure we have data
            if len(X) == 0 or len(y) == 0:
                logger.error("No samples available after processing")
                logger.info(f"X shape: {X.shape}, y shape: {len(y)}")
                return None, None

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

        # Check if we have enough samples for train_test_split
        min_samples = 10  # Minimum number of samples required
        if len(X) < min_samples:
            logger.error(f"Not enough samples to train model. Got {len(X)}, need at least {min_samples}")
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

            # Calculate entry/exit points with TP/SL levels
            self._calculate_trading_levels(df)

            logger.info(f"Made predictions for {len(df)} samples")
            return df

        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            return None

    def _calculate_trading_levels(self, df):
        """
        Calculate entry/exit points with stop-loss and take-profit levels.

        Args:
            df (pd.DataFrame): DataFrame with predictions
        """
        try:
            # Get the latest price and prediction
            latest_price = df['close'].iloc[-1]
            latest_pred = df['prediction'].iloc[-1]
            latest_prob = df.get('prediction_probability', pd.Series([0.5])).iloc[-1]

            # Calculate ATR (Average True Range) for dynamic SL/TP if available
            atr_value = None
            if 'atr_14' in df.columns:
                atr_value = df['atr_14'].iloc[-1]
            elif 'atr' in df.columns:
                atr_value = df['atr'].iloc[-1]

            # Default risk percentages
            sl_pct = 0.03  # 3% stop loss
            tp_pct = 0.05  # 5% take profit

            # Adjust based on prediction confidence
            confidence_factor = abs(latest_prob - 0.5) * 2  # Scale 0.5-1.0 to 0-1.0

            # If ATR is available, use it to calculate dynamic SL/TP
            if atr_value is not None and atr_value > 0:
                # Use ATR-based calculation with confidence adjustment
                atr_multiplier_sl = 2.0 + (1.0 * confidence_factor)  # 2-3x ATR for stop loss
                atr_multiplier_tp = 3.0 + (2.0 * confidence_factor)  # 3-5x ATR for take profit

                sl_amount = atr_value * atr_multiplier_sl
                tp_amount = atr_value * atr_multiplier_tp

                # Convert to percentage of price for consistency
                sl_pct = sl_amount / latest_price
                tp_pct = tp_amount / latest_price

            # Calculate entry, stop loss, and take profit based on prediction
            if latest_pred == 1:  # Bullish
                # Entry slightly below current price for better entry
                entry_price = latest_price * 0.995
                stop_loss = entry_price * (1 - sl_pct)
                take_profit = entry_price * (1 + tp_pct)
            else:  # Bearish
                # Entry slightly above current price for better entry
                entry_price = latest_price * 1.005
                stop_loss = entry_price * (1 + sl_pct)
                take_profit = entry_price * (1 - tp_pct)

            # Calculate risk-reward ratio
            risk_amount = abs(entry_price - stop_loss)
            reward_amount = abs(take_profit - entry_price)
            risk_reward = reward_amount / risk_amount if risk_amount > 0 else 0

            # Add to DataFrame
            df['entry_price'] = entry_price
            df['stop_loss'] = stop_loss
            df['take_profit'] = take_profit
            df['risk_reward'] = risk_reward

            logger.info(f"Calculated trading levels: Entry={entry_price:.2f}, SL={stop_loss:.2f}, TP={take_profit:.2f}, R/R={risk_reward:.2f}")

        except Exception as e:
            logger.error(f"Error calculating trading levels: {e}")
            # Set default values if calculation fails
            df['entry_price'] = df['close'].iloc[-1]
            df['stop_loss'] = df['close'].iloc[-1] * 0.97 if df['prediction'].iloc[-1] == 1 else df['close'].iloc[-1] * 1.03
            df['take_profit'] = df['close'].iloc[-1] * 1.05 if df['prediction'].iloc[-1] == 1 else df['close'].iloc[-1] * 0.95
            df['risk_reward'] = 1.67  # Default 1:1.67 risk-reward ratio


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
