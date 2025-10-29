# MediSense AI – Post-Surgical Recovery Prediction

## What This Is

I built this app to predict how long patients will take to recover after surgery. It's basically a machine learning tool that looks at patient info like age, BMI, pain level, etc. and gives you an estimate of recovery time in days.

I made this for a health hackathon I'm preparing for. The idea came from talking to some friends in med school who mentioned how hard it is to predict patient recovery times. I thought it would be cool to see if I could build something that helps with that.

## How I Built It

I started by researching what factors actually affect recovery time. Turns out there's a lot of research on this stuff. Then I built the whole thing step by step:

1. **Data Generation**: Since I can't use real patient data, I wrote code to create fake but realistic patient records. I made sure the relationships between different factors (like age and recovery time) actually make sense based on medical literature.

2. **Data Analysis**: I added tools to explore the data - correlation heatmaps, distribution plots, stuff like that. Helps you understand what's going on in the dataset.

3. **Machine Learning**: Used Random Forest because it's pretty reliable and gives good feature importance scores. Split the data properly and made sure to evaluate it correctly.

4. **Web Interface**: Built a Streamlit app so people can actually use it without needing to know Python. Made it look professional since it's for healthcare.

5. **Reproducibility**: Made sure everything uses the same random seeds so you get the same results every time. Important for research.

## Key Features

- **Local-only operation** - No cloud dependencies, no external data downloads
- **Synthetic data generation** - Realistic clinical data created locally
- **Complete ML pipeline** - From data generation to model evaluation
- **Interactive web interface** - User-friendly Streamlit application
- **Reproducible results** - Deterministic data generation and model training
- **Clinical context** - Designed for healthcare research applications

## Technical Implementation

### Data Generation
- Generate realistic synthetic clinical datasets with configurable sample sizes
- Deterministic data generation using random seeds for reproducibility
- Data validation and quality checks
- Clinical ranges and realistic distributions

### Exploratory Data Analysis
- Interactive correlation heatmaps
- Feature distribution visualizations
- Data quality assessment and reporting
- Statistical summaries and insights

### Machine Learning Pipeline
- Random Forest regression for recovery time prediction
- Comprehensive model evaluation metrics (R², MAE, RMSE)
- Feature importance analysis
- Predicted vs. actual performance visualization

### Interactive Prediction
- Single-patient prediction interface
- Real-time confidence estimates
- Clinical interpretation of results
- Input validation and error handling

### Export & Reproducibility
- Downloadable datasets in CSV format
- Comprehensive model reports in Markdown
- Reproducible parameter tracking
- Optional plot saving to local files

## Tech Stack

- **Python 3.10+** - Core programming language
- **Streamlit** - Web application framework
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computing
- **matplotlib** - Plotting and visualization
- **seaborn** - Statistical data visualization
- **scikit-learn** - Machine learning algorithms
- **joblib** - Model serialization

## How to Run Locally

### Prerequisites
- Python 3.10 or higher
- pip package manager

### Installation

1. **Clone or download the project:**
   ```bash
   # If using git
   git clone <repository-url>
   cd medisense_ai
   
   # Or simply download and extract the project files
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv medisense_env
   source medisense_env/bin/activate  # On Windows: medisense_env\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application:**
   ```bash
   streamlit run app.py
   ```

5. **Open your browser:**
   - The app will automatically open at `http://localhost:8501`
   - If it doesn't open automatically, navigate to the URL manually

### First-Time Setup Notes

- The application will create a `results/` folder for saving plots and reports
- All data is generated locally - no internet connection required
- The first run may take a moment to generate the initial dataset

## Usage Guide

### Step-by-Step Workflow

1. **Generate Data**
   - Use the sidebar controls to set sample size (100-2000) and random seed
   - Click "Generate Data" to create a synthetic clinical dataset
   - Or use "Auto-Generate & Train" for a quick start

2. **Explore the Data**
   - Review the data overview section for basic statistics
   - Click "Generate EDA Plots" to create correlation heatmaps and distributions
   - Examine data quality and validation results

3. **Train the Model**
   - Click "Train Model" to train the Random Forest regressor
   - View performance metrics (R², MAE, RMSE) with clinical interpretations
   - Examine feature importance rankings

4. **Try Predictions**
   - Use the "Interactive Prediction Demo" to test single-patient predictions
   - Input patient characteristics and get recovery time estimates
   - Review confidence intervals and clinical context

5. **Export Results**
   - Download the dataset as CSV for further analysis
   - Generate comprehensive model reports in Markdown format
   - Save plots to the `results/` folder (optional)

### Screenshots

*Screenshots will be added after first run:*
- Home screen with data generation controls
- Correlation heatmap showing feature relationships
- Feature importance visualization
- Predicted vs. actual scatter plot

## Data Schema

The synthetic clinical dataset contains the following features:

| Feature | Type | Range | Unit | Description |
|---------|------|-------|------|-------------|
| Age | Integer | 20-80 | years | Patient age at time of surgery |
| BMI | Float | 15-50 | kg/m² | Body Mass Index |
| Heart_Rate | Integer | 40-150 | bpm | Resting heart rate |
| Pain_Level | Integer | 1-10 | scale | Self-reported pain level (1=no pain, 10=severe) |
| Mobility_Score | Integer | 1-10 | scale | Physical mobility assessment (1=immobile, 10=fully mobile) |
| Recovery_Time | Float | 1-60 | days | Days to full recovery (target variable) |

### Target Variable Formula

The recovery time is generated using a clinically-informed formula:

```
Recovery_Time = Base(10) + Age_Factor + BMI_Factor + Pain_Factor + Mobility_Factor + HR_Factor + Noise

Where:
- Age_Factor = max(0, Age - 30) × 0.1
- BMI_Factor = max(0, BMI - 25) × 0.2
- Pain_Factor = Pain_Level × 1.0
- Mobility_Factor = -Mobility_Score × 0.5
- HR_Factor = max(0, Heart_Rate - 80) × 0.05
- Noise = Normal(0, 2) days
```

This formula reflects clinical intuition: older patients, higher BMI, more pain, and lower mobility lead to longer recovery times, while better mobility and lower heart rate contribute to faster recovery.

## Metrics & Interpretation

### Model Performance Metrics

- **R² Score (R-squared):** Measures the proportion of variance in recovery time explained by the model
  - 0.8+ = Excellent explanatory power
  - 0.6-0.8 = Good explanatory power
  - 0.4-0.6 = Moderate explanatory power
  - <0.4 = Limited explanatory power

- **MAE (Mean Absolute Error):** Average prediction error in days
  - ≤2 days = Very high precision
  - 2-4 days = High precision
  - 4-6 days = Moderate precision
  - >6 days = Limited precision

- **RMSE (Root Mean Square Error):** Square root of mean squared error, indicating typical prediction variability
  - ≤3 days = Low variability
  - 3-5 days = Moderate variability
  - >5 days = High variability

### Clinical Interpretation

The model's performance metrics should be interpreted in clinical context:

- **MAE of 2-4 days** is clinically useful for discharge planning and resource allocation
- **R² of 0.6+** indicates the model captures meaningful patterns in recovery time
- **Feature importance** helps identify which patient factors most influence recovery

## Limitations

### Data Limitations
- **Synthetic Data:** All data is artificially generated, not from real patients
- **Limited Features:** Only 5 clinical features considered; real models would include many more
- **No External Validation:** Model only evaluated on the same synthetic dataset
- **Simplified Relationships:** Real clinical relationships are more complex

### Model Limitations
- **Simple Algorithm:** Random Forest is basic; more sophisticated methods could improve performance
- **No Cross-Validation:** Model stability across different data splits not assessed
- **No Uncertainty Quantification:** No confidence intervals for individual predictions
- **No Temporal Modeling:** Does not account for recovery progression over time

### Clinical Considerations
- **Not for Clinical Use:** This is a demonstration model, not for actual patient care
- **Individual Variation:** Recovery time varies significantly between patients
- **Surgical Complexity:** Does not account for procedure-specific recovery patterns
- **Missing Factors:** Many important clinical factors not included

## Next Steps

### Immediate Improvements
1. **Real Data Integration:** Replace synthetic data with real, de-identified patient records
2. **Feature Engineering:** Add procedure type, comorbidities, and surgical complexity metrics
3. **Cross-Validation:** Implement k-fold cross-validation for robust performance estimation
4. **Hyperparameter Tuning:** Optimize model parameters using grid search or Bayesian optimization

### Advanced Enhancements
1. **Ensemble Methods:** Combine multiple algorithms (XGBoost, Neural Networks) for better performance
2. **Interpretability:** Add SHAP values for individual prediction explanations
3. **Uncertainty Quantification:** Implement prediction intervals and confidence measures
4. **Temporal Modeling:** Account for recovery progression over time
5. **Feature Selection:** Automated feature selection and engineering

### Production Considerations
1. **Model Validation:** Extensive validation on independent datasets
2. **Bias Assessment:** Evaluate model fairness across different patient populations
3. **Regulatory Compliance:** Ensure adherence to medical device regulations
4. **Clinical Integration:** Design user-friendly interfaces for healthcare providers
5. **Monitoring:** Implement model performance monitoring and drift detection

## How I Actually Built This

### Day 1: Research and Planning
- Spent way too much time reading medical papers about recovery factors
- Realized I needed to understand what actually affects recovery time
- Decided on the key features: age, BMI, heart rate, pain level, mobility
- Started sketching out the app structure

### Day 2: Data Stuff
- Built the synthetic data generator (this was harder than I thought)
- Had to make sure the fake data actually looked realistic
- Added validation to catch any weird data issues
- Spent a lot of time tweaking the formulas to get realistic relationships

### Day 3: Machine Learning
- Implemented Random Forest (chose it because it's reliable)
- Got the train/test split working properly
- Added feature importance so you can see what matters most
- Tried to make the model evaluation actually useful

### Day 4: Making It Usable
- Built the Streamlit interface (first time using it, actually pretty cool)
- Added all the visualization stuff
- Made the prediction form work properly
- Added export functionality because why not

### Day 5: Polish and Testing
- Tested everything to make sure it actually works
- Fixed a bunch of bugs (there were more than I expected)
- Wrote this README
- Got ready for the hackathon

## Why I Made This

I'm preparing for a health hackathon and wanted to build something that actually matters. After talking to some med students, I realized that predicting recovery times is a real problem in hospitals. 

This app shows what I can do with data science and machine learning. It's not perfect (obviously), but it demonstrates the whole pipeline from data generation to predictions. Hopefully it's useful enough to impress some judges.

**Important:** This is just a demo project. Don't use it for real medical decisions - I'm not a doctor and this isn't validated for clinical use. All the data is fake anyway.
