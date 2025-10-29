"""
Visualization utilities for MediSense AI.

This module provides functions for creating plots and visualizations
used in the EDA and model evaluation sections of the application.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict
import os
from constants import (
    CORRELATION_PLOT_PATH, FEATURE_IMPORTANCE_PLOT_PATH, 
    PRED_VS_ACTUAL_PLOT_PATH, RESULTS_DIR
)

# Set style for consistent, professional plots
plt.style.use('default')
sns.set_palette("husl")


def create_correlation_heatmap(df: pd.DataFrame, 
                              save_path: Optional[str] = None,
                              figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
    """
    Create correlation heatmap for numeric features.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset with numeric columns
    save_path : Optional[str]
        Path to save the plot (if None, plot is not saved)
    figsize : Tuple[int, int]
        Figure size (width, height)
        
    Returns:
    --------
    plt.Figure
        Matplotlib figure object
    """
    # Select only numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numeric_cols].corr()
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(
        corr_matrix, 
        mask=mask,
        annot=True, 
        cmap='RdBu_r', 
        center=0,
        square=True,
        fmt='.2f',
        cbar_kws={"shrink": .8},
        ax=ax
    )
    
    ax.set_title('Feature Correlation Matrix', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Features', fontsize=12)
    ax.set_ylabel('Features', fontsize=12)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def create_feature_distributions(df: pd.DataFrame, 
                                features: list = None,
                                save_path: Optional[str] = None,
                                figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
    """
    Create distribution plots for key features.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset
    features : list
        List of features to plot (if None, uses default features)
    save_path : Optional[str]
        Path to save the plot (if None, plot is not saved)
    figsize : Tuple[int, int]
        Figure size (width, height)
        
    Returns:
    --------
    plt.Figure
        Matplotlib figure object
    """
    if features is None:
        features = ['Age', 'BMI', 'Pain_Level', 'Mobility_Score', 'Recovery_Time']
    
    # Filter features that exist in the dataframe
    features = [f for f in features if f in df.columns]
    
    n_features = len(features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, feature in enumerate(features):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]
        
        if df[feature].dtype in ['int64', 'float64']:
            # For numeric features, create histogram
            ax.hist(df[feature], bins=20, alpha=0.7, edgecolor='black', linewidth=0.5)
            ax.set_xlabel(feature)
            ax.set_ylabel('Frequency')
            ax.set_title(f'Distribution of {feature}')
            
            # Add mean line
            mean_val = df[feature].mean()
            ax.axvline(mean_val, color='red', linestyle='--', alpha=0.8, 
                      label=f'Mean: {mean_val:.1f}')
            ax.legend()
        else:
            # For categorical features, create bar plot
            value_counts = df[feature].value_counts().sort_index()
            ax.bar(value_counts.index, value_counts.values, alpha=0.7, edgecolor='black')
            ax.set_xlabel(feature)
            ax.set_ylabel('Count')
            ax.set_title(f'Distribution of {feature}')
    
    # Hide empty subplots
    for i in range(n_features, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].set_visible(False)
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def create_feature_importance_plot(feature_importance: pd.Series,
                                  save_path: Optional[str] = None,
                                  figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
    """
    Create horizontal bar plot for feature importance.
    
    Parameters:
    -----------
    feature_importance : pd.Series
        Feature importance scores (index=feature_names, values=importance)
    save_path : Optional[str]
        Path to save the plot (if None, plot is not saved)
    figsize : Tuple[int, int]
        Figure size (width, height)
        
    Returns:
    --------
    plt.Figure
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create horizontal bar plot
    bars = ax.barh(range(len(feature_importance)), feature_importance.values, 
                   color='skyblue', alpha=0.8, edgecolor='navy', linewidth=0.5)
    
    # Customize the plot
    ax.set_yticks(range(len(feature_importance)))
    ax.set_yticklabels(feature_importance.index)
    ax.set_xlabel('Feature Importance', fontsize=12)
    ax.set_title('Feature Importance for Recovery Time Prediction', 
                fontsize=14, fontweight='bold', pad=20)
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, feature_importance.values)):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{value:.3f}', ha='left', va='center', fontsize=10)
    
    # Add grid for better readability
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def create_predicted_vs_actual_plot(y_actual: np.ndarray, 
                                   y_predicted: np.ndarray,
                                   save_path: Optional[str] = None,
                                   figsize: Tuple[int, int] = (8, 8)) -> plt.Figure:
    """
    Create predicted vs actual scatter plot with reference line.
    
    Parameters:
    -----------
    y_actual : np.ndarray
        Actual target values
    y_predicted : np.ndarray
        Predicted target values
    save_path : Optional[str]
        Path to save the plot (if None, plot is not saved)
    figsize : Tuple[int, int]
        Figure size (width, height)
        
    Returns:
    --------
    plt.Figure
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create scatter plot
    ax.scatter(y_actual, y_predicted, alpha=0.6, s=50, color='steelblue', 
               edgecolors='navy', linewidth=0.5)
    
    # Add perfect prediction line (y=x)
    min_val = min(y_actual.min(), y_predicted.min())
    max_val = max(y_actual.max(), y_predicted.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, 
            label='Perfect Prediction (y=x)', alpha=0.8)
    
    # Calculate R² for annotation
    from sklearn.metrics import r2_score
    r2 = r2_score(y_actual, y_predicted)
    
    # Add R² annotation
    ax.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax.transAxes, 
            fontsize=12, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Customize the plot
    ax.set_xlabel('Actual Recovery Time (days)', fontsize=12)
    ax.set_ylabel('Predicted Recovery Time (days)', fontsize=12)
    ax.set_title('Predicted vs Actual Recovery Time', fontsize=14, fontweight='bold', pad=20)
    ax.legend()
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Set equal aspect ratio
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def create_residuals_plot(y_actual: np.ndarray, 
                         y_predicted: np.ndarray,
                         save_path: Optional[str] = None,
                         figsize: Tuple[int, int] = (12, 5)) -> plt.Figure:
    """
    Create residuals analysis plots.
    
    Parameters:
    -----------
    y_actual : np.ndarray
        Actual target values
    y_predicted : np.ndarray
        Predicted target values
    save_path : Optional[str]
        Path to save the plot (if None, plot is not saved)
    figsize : Tuple[int, int]
        Figure size (width, height)
        
    Returns:
    --------
    plt.Figure
        Matplotlib figure object
    """
    residuals = y_actual - y_predicted
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Residuals vs Predicted
    ax1.scatter(y_predicted, residuals, alpha=0.6, s=50, color='steelblue')
    ax1.axhline(y=0, color='red', linestyle='--', linewidth=2)
    ax1.set_xlabel('Predicted Recovery Time (days)')
    ax1.set_ylabel('Residuals (Actual - Predicted)')
    ax1.set_title('Residuals vs Predicted')
    ax1.grid(True, alpha=0.3)
    
    # Residuals histogram
    ax2.hist(residuals, bins=20, alpha=0.7, color='skyblue', edgecolor='navy')
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax2.set_xlabel('Residuals (Actual - Predicted)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of Residuals')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def save_all_plots(df: pd.DataFrame, 
                  feature_importance: pd.Series,
                  y_actual: np.ndarray,
                  y_predicted: np.ndarray) -> Dict[str, str]:
    """
    Save all plots to the results directory.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset for correlation and distribution plots
    feature_importance : pd.Series
        Feature importance scores
    y_actual : np.ndarray
        Actual target values
    y_predicted : np.ndarray
        Predicted target values
        
    Returns:
    --------
    Dict[str, str]
        Dictionary mapping plot names to file paths
    """
    # Ensure results directory exists
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    saved_plots = {}
    
    # Save correlation heatmap
    fig_corr = create_correlation_heatmap(df, CORRELATION_PLOT_PATH)
    plt.close(fig_corr)
    saved_plots['correlation'] = CORRELATION_PLOT_PATH
    
    # Save feature distributions
    fig_dist = create_feature_distributions(df, save_path=f"{RESULTS_DIR}/feature_distributions.png")
    plt.close(fig_dist)
    saved_plots['distributions'] = f"{RESULTS_DIR}/feature_distributions.png"
    
    # Save feature importance
    fig_imp = create_feature_importance_plot(feature_importance, FEATURE_IMPORTANCE_PLOT_PATH)
    plt.close(fig_imp)
    saved_plots['feature_importance'] = FEATURE_IMPORTANCE_PLOT_PATH
    
    # Save predicted vs actual
    fig_pred = create_predicted_vs_actual_plot(y_actual, y_predicted, PRED_VS_ACTUAL_PLOT_PATH)
    plt.close(fig_pred)
    saved_plots['predicted_vs_actual'] = PRED_VS_ACTUAL_PLOT_PATH
    
    # Save residuals plot
    fig_res = create_residuals_plot(y_actual, y_predicted, save_path=f"{RESULTS_DIR}/residuals.png")
    plt.close(fig_res)
    saved_plots['residuals'] = f"{RESULTS_DIR}/residuals.png"
    
    return saved_plots
