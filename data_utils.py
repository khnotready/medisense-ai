"""
Data generation and validation for MediSense AI.

This handles creating fake patient data and making sure it's valid.
I tried to make the fake data realistic based on medical research.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, Optional
import warnings
from constants import (
    COLUMNS, FEATURE_COLUMNS, TARGET_COLUMN, VALIDATION_RANGES,
    DEFAULT_SAMPLE_SIZE, MIN_SAMPLE_SIZE, MAX_SAMPLE_SIZE
)


def generate_synthetic_data(n_samples: int = DEFAULT_SAMPLE_SIZE, 
                          seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic clinical dataset for post-surgical recovery prediction.
    
    This function creates realistic synthetic data that mimics real clinical
    patterns while ensuring reproducibility through deterministic generation.
    
    Parameters:
    -----------
    n_samples : int
        Number of patients to generate (100-2000)
    seed : int
        Random seed for reproducibility
        
    Returns:
    --------
    pd.DataFrame
        Synthetic clinical dataset with columns: Age, BMI, Heart_Rate, 
        Pain_Level, Mobility_Score, Recovery_Time
        
    Notes:
    ------
    The recovery time formula is based on clinical intuition:
    - Base recovery: 7-14 days for healthy patients
    - Age factor: +0.1 days per year over 30
    - BMI factor: +0.2 days per BMI point over 25
    - Pain factor: +1 day per pain level point
    - Mobility factor: -0.5 days per mobility score point
    - Heart rate factor: +0.05 days per bpm over 80
    - Random noise: ±2-5 days for realistic variation
    """
    # Clamp sample size to valid range
    n_samples = max(MIN_SAMPLE_SIZE, min(MAX_SAMPLE_SIZE, n_samples))
    
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    # Generate base features with realistic distributions
    data = {}
    
    # Age: slightly skewed towards older patients (typical surgical population)
    data['Age'] = np.random.gamma(2, 15, n_samples) + 20
    data['Age'] = np.clip(data['Age'], 20, 80)
    
    # BMI: normal distribution around 27 (typical for surgical patients)
    data['BMI'] = np.random.normal(27, 4, n_samples)
    data['BMI'] = np.clip(data['BMI'], 15, 50)
    
    # Heart Rate: normal distribution around 75 bpm
    data['Heart_Rate'] = np.random.normal(75, 12, n_samples)
    data['Heart_Rate'] = np.clip(data['Heart_Rate'], 40, 150)
    
    # Pain Level: discrete values 1-10, weighted towards lower values
    pain_weights = [0.05, 0.1, 0.15, 0.2, 0.2, 0.15, 0.1, 0.03, 0.015, 0.005]
    data['Pain_Level'] = np.random.choice(range(1, 11), n_samples, p=pain_weights)
    
    # Mobility Score: discrete values 1-10, weighted towards higher values
    mobility_weights = [0.01, 0.02, 0.05, 0.08, 0.12, 0.18, 0.22, 0.18, 0.1, 0.04]
    data['Mobility_Score'] = np.random.choice(range(1, 11), n_samples, p=mobility_weights)
    
    # Calculate Recovery Time using clinical formula + noise
    base_recovery = 10  # Base 10 days
    
    # Age factor: +0.1 days per year over 30
    age_factor = np.maximum(0, data['Age'] - 30) * 0.1
    
    # BMI factor: +0.2 days per BMI point over 25
    bmi_factor = np.maximum(0, data['BMI'] - 25) * 0.2
    
    # Pain factor: +1 day per pain level point
    pain_factor = data['Pain_Level'] * 1.0
    
    # Mobility factor: -0.5 days per mobility score point (better mobility = faster recovery)
    mobility_factor = -data['Mobility_Score'] * 0.5
    
    # Heart rate factor: +0.05 days per bpm over 80
    hr_factor = np.maximum(0, data['Heart_Rate'] - 80) * 0.05
    
    # Random noise: ±2-5 days for realistic variation
    noise = np.random.normal(0, 2, n_samples)
    
    # Calculate final recovery time
    data['Recovery_Time'] = (base_recovery + age_factor + bmi_factor + 
                           pain_factor + mobility_factor + hr_factor + noise)
    
    # Ensure positive recovery times and reasonable range
    data['Recovery_Time'] = np.clip(data['Recovery_Time'], 1, 60)
    
    # Create DataFrame
    df = pd.DataFrame(data, columns=COLUMNS)
    
    # Round numeric columns to appropriate precision
    df['Age'] = df['Age'].round(0).astype(int)
    df['BMI'] = df['BMI'].round(1)
    df['Heart_Rate'] = df['Heart_Rate'].round(0).astype(int)
    df['Recovery_Time'] = df['Recovery_Time'].round(1)
    
    return df


def validate_data(df: pd.DataFrame) -> Tuple[bool, Dict[str, Any]]:
    """
    Validate synthetic dataset for quality and range compliance.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset to validate
        
    Returns:
    --------
    Tuple[bool, Dict[str, Any]]
        (is_valid, validation_report) where validation_report contains
        warnings, missing values, and range violations
    """
    validation_report = {
        'warnings': [],
        'missing_values': df.isnull().sum().to_dict(),
        'range_violations': {},
        'shape': df.shape,
        'dtypes': df.dtypes.to_dict()
    }
    
    is_valid = True
    
    # Check for missing values
    if df.isnull().sum().sum() > 0:
        validation_report['warnings'].append("Dataset contains missing values")
        is_valid = False
    
    # Check data types
    expected_types = {
        'Age': 'int64',
        'BMI': 'float64', 
        'Heart_Rate': 'int64',
        'Pain_Level': 'int64',
        'Mobility_Score': 'int64',
        'Recovery_Time': 'float64'
    }
    
    for col, expected_type in expected_types.items():
        if col in df.columns and str(df[col].dtype) != expected_type:
            validation_report['warnings'].append(f"Column {col} has unexpected dtype: {df[col].dtype}")
    
    # Check value ranges
    for col, (min_val, max_val) in VALIDATION_RANGES.items():
        if col in df.columns:
            violations = df[(df[col] < min_val) | (df[col] > max_val)]
            if len(violations) > 0:
                validation_report['range_violations'][col] = {
                    'count': len(violations),
                    'min_found': df[col].min(),
                    'max_found': df[col].max(),
                    'expected_range': (min_val, max_val)
                }
                validation_report['warnings'].append(
                    f"Column {col} has {len(violations)} values outside expected range ({min_val}-{max_val})"
                )
                is_valid = False
    
    return is_valid, validation_report


def get_data_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate a comprehensive summary of the dataset for display in the UI.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset to summarize
        
    Returns:
    --------
    Dict[str, Any]
        Summary statistics and metadata
    """
    summary = {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'numeric_summary': df.describe().to_dict(),
        'categorical_summary': {}
    }
    
    # Add categorical summaries for discrete variables
    for col in ['Pain_Level', 'Mobility_Score']:
        if col in df.columns:
            summary['categorical_summary'][col] = df[col].value_counts().sort_index().to_dict()
    
    return summary
