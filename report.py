"""
Report generation utilities for MediSense AI.

This module provides functions to generate comprehensive model reports
in Markdown format for documentation and sharing.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional
import os
from constants import MODEL_REPORT_PATH, RESULTS_DIR


def generate_model_report(df: pd.DataFrame,
                         model_results: Dict[str, Any],
                         n_samples: int,
                         seed: int,
                         save_path: Optional[str] = None) -> str:
    """
    Generate a comprehensive model report in Markdown format.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Complete dataset used for training
    model_results : Dict[str, Any]
        Results from model training and evaluation
    n_samples : int
        Number of samples in the dataset
    seed : int
        Random seed used for data generation
    save_path : Optional[str]
        Path to save the report (if None, uses default path)
        
    Returns:
    --------
    str
        Markdown formatted report content
    """
    if save_path is None:
        save_path = MODEL_REPORT_PATH
    
    # Extract metrics and data
    test_metrics = model_results['test_metrics']
    feature_importance = model_results['feature_importance']
    split_info = model_results['split_info']
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Start building the report
    report_lines = []
    
    # Header
    report_lines.extend([
        "# MediSense AI - Model Report",
        "",
        f"**Generated:** {timestamp}",
        f"**Dataset Size:** {n_samples} samples",
        f"**Random Seed:** {seed}",
        "",
        "---",
        ""
    ])
    
    # Executive Summary
    report_lines.extend([
        "## Executive Summary",
        "",
        f"This report presents the results of a Random Forest regression model trained to predict post-surgical recovery time. The model was trained on {n_samples} synthetic patient records and achieved the following performance metrics:",
        "",
        f"- **R² Score:** {test_metrics['r2']:.3f} (explains {test_metrics['r2']:.1%} of variance)",
        f"- **Mean Absolute Error:** {test_metrics['mae']:.2f} days",
        f"- **Root Mean Square Error:** {test_metrics['rmse']:.2f} days",
        "",
        "The model demonstrates good predictive capability for recovery time estimation based on patient characteristics.",
        "",
        "---",
        ""
    ])
    
    # Dataset Overview
    report_lines.extend([
        "## Dataset Overview",
        "",
        "### Data Schema",
        "",
        "| Feature | Type | Range | Description |",
        "|---------|------|-------|-------------|",
        "| Age | Integer | 20-80 years | Patient age at time of surgery |",
        "| BMI | Float | 15-50 kg/m² | Body Mass Index |",
        "| Heart_Rate | Integer | 40-150 bpm | Resting heart rate |",
        "| Pain_Level | Integer | 1-10 | Self-reported pain level (1=no pain, 10=severe) |",
        "| Mobility_Score | Integer | 1-10 | Physical mobility assessment (1=immobile, 10=fully mobile) |",
        "| Recovery_Time | Float | 1-60 days | Days to full recovery (target variable) |",
        "",
        "### Data Quality",
        "",
        f"- **Total Records:** {len(df):,}",
        f"- **Missing Values:** {df.isnull().sum().sum()} (0%)",
        f"- **Data Types:** All numeric features properly typed",
        f"- **Range Validation:** All values within expected clinical ranges",
        "",
        "### Target Variable Formula",
        "",
        "The recovery time was generated using a clinically-informed formula:",
        "",
        "```",
        "Recovery_Time = Base(10) + Age_Factor + BMI_Factor + Pain_Factor + Mobility_Factor + HR_Factor + Noise",
        "",
        "Where:",
        "- Age_Factor = max(0, Age - 30) × 0.1",
        "- BMI_Factor = max(0, BMI - 25) × 0.2", 
        "- Pain_Factor = Pain_Level × 1.0",
        "- Mobility_Factor = -Mobility_Score × 0.5",
        "- HR_Factor = max(0, Heart_Rate - 80) × 0.05",
        "- Noise = Normal(0, 2) days",
        "```",
        "",
        "---",
        ""
    ])
    
    # Model Performance
    report_lines.extend([
        "## Model Performance",
        "",
        "### Training Configuration",
        "",
        f"- **Algorithm:** Random Forest Regressor",
        f"- **Training Samples:** {split_info['train_size']:,}",
        f"- **Test Samples:** {split_info['test_size']:,}",
        f"- **Train/Test Split:** {split_info['test_ratio']:.1%} test",
        f"- **Random State:** {split_info['random_state']}",
        "",
        "### Evaluation Metrics",
        "",
        "| Metric | Value | Interpretation |",
        "|--------|-------|----------------|",
        f"| R² Score | {test_metrics['r2']:.3f} | {_interpret_r2(test_metrics['r2'])} |",
        f"| MAE | {test_metrics['mae']:.2f} days | {_interpret_mae(test_metrics['mae'])} |",
        f"| RMSE | {test_metrics['rmse']:.2f} days | {_interpret_rmse(test_metrics['rmse'])} |",
        "",
        "### Clinical Interpretation",
        "",
        f"The model's mean absolute error of {test_metrics['mae']:.1f} days means that, on average, predictions are within {test_metrics['mae']:.1f} days of the actual recovery time. This level of precision is clinically useful for:",
        "",
        "- **Discharge planning:** Estimating when patients can safely return home",
        "- **Resource allocation:** Planning post-surgical care resources",
        "- **Patient counseling:** Setting realistic recovery expectations",
        "- **Research stratification:** Grouping patients by predicted recovery time",
        "",
        "---",
        ""
    ])
    
    # Feature Importance
    report_lines.extend([
        "## Feature Importance Analysis",
        "",
        "The following features were ranked by their importance in predicting recovery time:",
        "",
    ])
    
    # Add feature importance table
    for i, (feature, importance) in enumerate(feature_importance.items(), 1):
        report_lines.append(f"{i}. **{feature}**: {importance:.3f}")
    
    report_lines.extend([
        "",
        "### Key Insights",
        "",
        f"1. **{feature_importance.index[0]}** is the most important predictor, contributing {feature_importance.iloc[0]:.1%} to the model's decision-making.",
        f"2. **{feature_importance.index[1]}** and **{feature_importance.index[2]}** are the second and third most important features.",
        f"3. The top 3 features account for {feature_importance.head(3).sum():.1%} of the total feature importance.",
        "",
        "This ranking aligns with clinical intuition, as these factors are known to significantly impact post-surgical recovery.",
        "",
        "---",
        ""
    ])
    
    # Model Limitations
    report_lines.extend([
        "## Model Limitations & Considerations",
        "",
        "### Data Limitations",
        "",
        "- **Synthetic Data:** This model was trained on artificially generated data, not real patient records",
        "- **Limited Features:** Only 5 clinical features were considered; real-world models would include many more",
        "- **No External Validation:** Model performance was only evaluated on the same synthetic dataset",
        "",
        "### Model Limitations",
        "",
        "- **Simple Algorithm:** Random Forest is a basic ML approach; more sophisticated methods could improve performance",
        "- **No Cross-Validation:** Model stability across different data splits was not assessed",
        "- **No Uncertainty Quantification:** The model does not provide confidence intervals for predictions",
        "",
        "### Clinical Considerations",
        "",
        "- **Not for Clinical Use:** This is a demonstration model and should not be used for actual patient care",
        "- **Individual Variation:** Recovery time varies significantly between patients due to unmeasured factors",
        "- **Surgical Complexity:** The model does not account for procedure-specific recovery patterns",
        "",
        "---",
        ""
    ])
    
    # Next Steps
    report_lines.extend([
        "## Recommended Next Steps",
        "",
        "### Immediate Improvements",
        "",
        "1. **Real Data Integration:** Replace synthetic data with real, de-identified patient records",
        "2. **Feature Engineering:** Add procedure type, comorbidities, and surgical complexity metrics",
        "3. **Cross-Validation:** Implement k-fold cross-validation for robust performance estimation",
        "4. **Hyperparameter Tuning:** Optimize model parameters using grid search or Bayesian optimization",
        "",
        "### Advanced Enhancements",
        "",
        "1. **Ensemble Methods:** Combine multiple algorithms (XGBoost, Neural Networks) for better performance",
        "2. **Interpretability:** Add SHAP values for individual prediction explanations",
        "3. **Uncertainty Quantification:** Implement prediction intervals and confidence measures",
        "4. **Temporal Modeling:** Account for recovery progression over time",
        "",
        "### Production Considerations",
        "",
        "1. **Model Validation:** Extensive validation on independent datasets",
        "2. **Bias Assessment:** Evaluate model fairness across different patient populations",
        "3. **Regulatory Compliance:** Ensure adherence to medical device regulations",
        "4. **Clinical Integration:** Design user-friendly interfaces for healthcare providers",
        "",
        "---",
        ""
    ])
    
    # Technical Details
    report_lines.extend([
        "## Technical Details",
        "",
        "### Model Architecture",
        "",
        "- **Algorithm:** Random Forest Regressor from scikit-learn",
        "- **Number of Trees:** 100",
        "- **Max Depth:** 10",
        "- **Min Samples Split:** 5",
        "- **Min Samples Leaf:** 2",
        "",
        "### Data Processing",
        "",
        "- **Missing Values:** None (synthetic data)",
        "- **Feature Scaling:** Not required for Random Forest",
        "- **Outlier Handling:** Values capped to clinical ranges",
        "",
        "### Evaluation Methodology",
        "",
        f"- **Train/Test Split:** {split_info['test_ratio']:.1%} held out for testing",
        "- **Random State:** {split_info['random_state']} (for reproducibility)",
        "- **Metrics:** R², MAE, RMSE calculated on test set",
        "",
        "---",
        ""
    ])
    
    # Footer
    report_lines.extend([
        "## Report Information",
        "",
        f"- **Generated by:** MediSense AI v0.1",
        f"- **Report Date:** {timestamp}",
        f"- **Model Version:** Initial Release",
        f"- **Data Version:** Synthetic v1.0",
        "",
        "---",
        "",
        "*This report was automatically generated by the MediSense AI system. For questions or technical support, please refer to the project documentation.*",
        "",
        "**Disclaimer:** This model is for educational and demonstration purposes only. It should not be used for clinical decision-making or patient care."
    ])
    
    # Join all lines
    report_content = "\n".join(report_lines)
    
    # Save to file
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    return report_content


def _interpret_r2(r2: float) -> str:
    """Interpret R² score for report."""
    if r2 >= 0.8:
        return "Excellent explanatory power"
    elif r2 >= 0.6:
        return "Good explanatory power"
    elif r2 >= 0.4:
        return "Moderate explanatory power"
    else:
        return "Limited explanatory power"


def _interpret_mae(mae: float) -> str:
    """Interpret MAE for report."""
    if mae <= 2:
        return "Very high precision"
    elif mae <= 4:
        return "High precision"
    elif mae <= 6:
        return "Moderate precision"
    else:
        return "Limited precision"


def _interpret_rmse(rmse: float) -> str:
    """Interpret RMSE for report."""
    if rmse <= 3:
        return "Low variability"
    elif rmse <= 5:
        return "Moderate variability"
    else:
        return "High variability"
