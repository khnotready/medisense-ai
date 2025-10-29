"""
Machine learning pipeline and evaluation for MediSense AI.

This module provides functions for training, evaluating, and using the
post-surgical recovery time prediction model.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from typing import Tuple, Dict, Any, List
import joblib
from constants import (
    FEATURE_COLUMNS, TARGET_COLUMN, TRAIN_TEST_SPLIT, 
    RANDOM_STATE, N_ESTIMATORS
)


class RecoveryTimePredictor:
    """
    Random Forest model for predicting post-surgical recovery time.
    
    This class encapsulates the complete ML pipeline including training,
    evaluation, and prediction functionality.
    """
    
    def __init__(self, n_estimators: int = N_ESTIMATORS, random_state: int = RANDOM_STATE):
        """
        Initialize the recovery time predictor.
        
        Parameters:
        -----------
        n_estimators : int
            Number of trees in the random forest
        random_state : int
            Random seed for reproducibility
        """
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2
        )
        self.feature_columns = FEATURE_COLUMNS
        self.target_column = TARGET_COLUMN
        self.is_trained = False
        self.feature_importance_ = None
        self.training_metrics = None
        
    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Train the recovery time prediction model.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Feature matrix
        y : pd.Series
            Target variable (recovery time)
            
        Returns:
        --------
        Dict[str, float]
            Training metrics (R², MAE, RMSE)
        """
        # Ensure we have the correct feature columns
        X_train = X[self.feature_columns].copy()
        
        # Train the model
        self.model.fit(X_train, y)
        
        # Store feature importance
        self.feature_importance_ = pd.Series(
            self.model.feature_importances_,
            index=self.feature_columns
        ).sort_values(ascending=True)
        
        # Calculate training metrics
        y_pred = self.model.predict(X_train)
        
        self.training_metrics = {
            'r2_train': r2_score(y, y_pred),
            'mae_train': mean_absolute_error(y, y_pred),
            'rmse_train': np.sqrt(mean_squared_error(y, y_pred))
        }
        
        self.is_trained = True
        return self.training_metrics
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Evaluate the trained model on test data.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Test feature matrix
        y : pd.Series
            Test target variable
            
        Returns:
        --------
        Dict[str, float]
            Evaluation metrics (R², MAE, RMSE)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        X_test = X[self.feature_columns].copy()
        y_pred = self.model.predict(X_test)
        
        metrics = {
            'r2': r2_score(y, y_pred),
            'mae': mean_absolute_error(y, y_pred),
            'rmse': np.sqrt(mean_squared_error(y, y_pred))
        }
        
        return metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on new data.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Feature matrix for prediction
            
        Returns:
        --------
        np.ndarray
            Predicted recovery times
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        X_pred = X[self.feature_columns].copy()
        return self.model.predict(X_pred)
    
    def get_feature_importance(self) -> pd.Series:
        """
        Get feature importance scores.
        
        Returns:
        --------
        pd.Series
            Feature importance scores sorted in descending order
        """
        if not self.is_trained:
            raise ValueError("Model must be trained to get feature importance")
        
        return self.feature_importance_.sort_values(ascending=False)
    
    def get_residuals(self, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
        """
        Calculate prediction residuals.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Feature matrix
        y : pd.Series
            Actual target values
            
        Returns:
        --------
        np.ndarray
            Prediction residuals (actual - predicted)
        """
        y_pred = self.predict(X)
        return y - y_pred


def train_test_split_data(df: pd.DataFrame, 
                         test_size: float = 1 - TRAIN_TEST_SPLIT,
                         random_state: int = RANDOM_STATE) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split dataset into training and testing sets.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Complete dataset
    test_size : float
        Proportion of data to use for testing
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
        (X_train, X_test, y_train, y_test)
    """
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    return X_train, X_test, y_train, y_test


def train_model(df: pd.DataFrame, 
                test_size: float = 1 - TRAIN_TEST_SPLIT,
                random_state: int = RANDOM_STATE) -> Tuple[RecoveryTimePredictor, Dict[str, Any]]:
    """
    Complete model training pipeline.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Training dataset
    test_size : float
        Proportion of data to use for testing
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    Tuple[RecoveryTimePredictor, Dict[str, Any]]
        (trained_model, results_dict) containing model and evaluation metrics
    """
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split_data(df, test_size, random_state)
    
    # Initialize and train model
    model = RecoveryTimePredictor(random_state=random_state)
    train_metrics = model.train(X_train, y_train)
    
    # Evaluate on test set
    test_metrics = model.evaluate(X_test, y_test)
    
    # Get predictions for analysis
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate residuals
    train_residuals = model.get_residuals(X_train, y_train)
    test_residuals = model.get_residuals(X_test, y_test)
    
    # Compile results
    results = {
        'model': model,
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'feature_importance': model.get_feature_importance(),
        'y_train': y_train,
        'y_test': y_test,
        'y_train_pred': y_train_pred,
        'y_test_pred': y_test_pred,
        'train_residuals': train_residuals,
        'test_residuals': test_residuals,
        'split_info': {
            'train_size': len(X_train),
            'test_size': len(X_test),
            'test_ratio': test_size,
            'random_state': random_state
        }
    }
    
    return model, results


def interpret_metrics(metrics: Dict[str, float]) -> Dict[str, str]:
    """
    Provide human-readable interpretations of model metrics.
    
    Parameters:
    -----------
    metrics : Dict[str, float]
        Model evaluation metrics
        
    Returns:
    --------
    Dict[str, str]
        Interpretations of each metric for clinical context
    """
    interpretations = {}
    
    # R² interpretation
    r2 = metrics.get('r2', 0)
    if r2 >= 0.8:
        interpretations['r2'] = f"Excellent! The model explains {r2:.1%} of recovery time variation."
    elif r2 >= 0.6:
        interpretations['r2'] = f"Good! The model explains {r2:.1%} of recovery time variation."
    elif r2 >= 0.4:
        interpretations['r2'] = f"Moderate. The model explains {r2:.1%} of recovery time variation."
    else:
        interpretations['r2'] = f"Limited. The model explains only {r2:.1%} of recovery time variation."
    
    # MAE interpretation
    mae = metrics.get('mae', 0)
    if mae <= 2:
        interpretations['mae'] = f"Very precise! Average prediction error is {mae:.1f} days."
    elif mae <= 4:
        interpretations['mae'] = f"Good precision. Average prediction error is {mae:.1f} days."
    elif mae <= 6:
        interpretations['mae'] = f"Moderate precision. Average prediction error is {mae:.1f} days."
    else:
        interpretations['mae'] = f"Limited precision. Average prediction error is {mae:.1f} days."
    
    # RMSE interpretation
    rmse = metrics.get('rmse', 0)
    interpretations['rmse'] = f"Root mean square error is {rmse:.1f} days, indicating typical prediction variability."
    
    return interpretations


def get_confidence_interval(mae: float, n_samples: int) -> str:
    """
    Generate a confidence statement based on model performance.
    
    Parameters:
    -----------
    mae : float
        Mean absolute error of the model
    n_samples : int
        Number of samples used for evaluation
        
    Returns:
    --------
    str
        Confidence statement for UI display
    """
    if mae <= 2:
        confidence = "High confidence"
        range_text = f"±{mae:.1f} days"
    elif mae <= 4:
        confidence = "Moderate confidence"
        range_text = f"±{mae:.1f} days"
    else:
        confidence = "Lower confidence"
        range_text = f"±{mae:.1f} days"
    
    return f"{confidence}: Typical prediction error is {range_text} based on {n_samples} test samples."
