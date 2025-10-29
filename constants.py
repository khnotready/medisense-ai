"""
Constants and configuration for MediSense AI project.

This module centralizes all configuration values, column names, and file paths
used throughout the application to ensure consistency and maintainability.
"""

# Dataset configuration
DEFAULT_SAMPLE_SIZE = 300
MIN_SAMPLE_SIZE = 100
MAX_SAMPLE_SIZE = 2000
DEFAULT_SEED = 42

# Train/test split ratio
TRAIN_TEST_SPLIT = 0.8

# Column names for the synthetic clinical dataset
COLUMNS = [
    "Age",
    "BMI", 
    "Heart_Rate",
    "Pain_Level",
    "Mobility_Score",
    "Recovery_Time"
]

# Feature columns (excluding target)
FEATURE_COLUMNS = [
    "Age",
    "BMI",
    "Heart_Rate", 
    "Pain_Level",
    "Mobility_Score"
]

# Target column
TARGET_COLUMN = "Recovery_Time"

# Data validation ranges
VALIDATION_RANGES = {
    "Age": (20, 80),
    "BMI": (15, 50),
    "Heart_Rate": (40, 150),
    "Pain_Level": (1, 10),
    "Mobility_Score": (1, 10),
    "Recovery_Time": (1, 60)
}

# File paths
RESULTS_DIR = "results"
CORRELATION_PLOT_PATH = f"{RESULTS_DIR}/corr_heatmap.png"
FEATURE_IMPORTANCE_PLOT_PATH = f"{RESULTS_DIR}/feature_importance.png"
PRED_VS_ACTUAL_PLOT_PATH = f"{RESULTS_DIR}/pred_vs_actual.png"
MODEL_REPORT_PATH = f"{RESULTS_DIR}/model_report.md"
DATASET_CSV_PATH = f"{RESULTS_DIR}/synthetic_dataset.csv"

# Model configuration
RANDOM_STATE = 42
N_ESTIMATORS = 100

# UI Configuration
APP_TITLE = "🏥 MediSense AI — Post-Surgical Recovery Predictor (Local)"
APP_SUBTITLE = "Hackathon prototype • Synthetic data • No patient info • Offline only"
FOOTER_TEXT = "Built in 48 hours • Local only • Educational demo"

