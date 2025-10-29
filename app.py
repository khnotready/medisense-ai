import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime

# Add the current directory to the path so we can import our modules
sys.path.append('.')

from constants import *
from data_utils import generate_synthetic_data, get_data_summary, validate_data
from model import train_model
from visuals import create_correlation_heatmap, create_feature_distributions, create_feature_importance_plot, create_predicted_vs_actual_plot
from report import generate_model_report

# Page configuration
st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Styling to make it look professional
st.markdown("""
<style>
    /* Main header styling */
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        color: white;
        border-radius: 12px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 25px rgba(59, 130, 246, 0.15);
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .main-header h3 {
        margin: 0.5rem 0 0 0;
        font-size: 1.2rem;
        font-weight: 300;
        opacity: 0.95;
    }
    
    /* Section headers */
    .section-header {
        background: linear-gradient(135deg, #1e40af 0%, #3b82f6 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 8px;
        margin: 2rem 0 1rem 0;
        font-size: 1.3rem;
        font-weight: 600;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.2);
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #3b82f6;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.1);
        transition: all 0.3s ease;
        color: #1e293b;
        border: 1px solid #e2e8f0;
    }
    
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(59, 130, 246, 0.15);
        border-left-color: #1e40af;
    }
    
    /* Status boxes */
    .warning-box {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        border: 1px solid #f59e0b;
        border-left: 4px solid #f59e0b;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(245, 158, 11, 0.1);
        color: #92400e;
    }
    
    .success-box {
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
        border: 1px solid #10b981;
        border-left: 4px solid #10b981;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.1);
        color: #065f46;
    }
    
    .info-box {
        background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
        border: 1px solid #3b82f6;
        border-left: 4px solid #3b82f6;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.1);
        color: #1e40af;
    }
    
    /* Buttons styling */
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #1e40af 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.8rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
        font-size: 0.95rem;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(59, 130, 246, 0.4);
        background: linear-gradient(135deg, #1e40af 0%, #1e3a8a 100%);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%);
        border-right: 1px solid #e2e8f0;
    }
    
    /* Dataframe styling */
    .dataframe {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        border: 1px solid #e2e8f0;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        color: white;
        margin-top: 3rem;
        border-radius: 12px;
        font-size: 0.9rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }
    
    /* Feature cards */
    .feature-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        margin: 1rem 0;
        border-left: 4px solid #3b82f6;
        transition: all 0.3s ease;
        color: #1e293b;
        border: 1px solid #e2e8f0;
    }
    
    .feature-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(59, 130, 246, 0.1);
        border-left-color: #1e40af;
    }
    
    /* Progress container */
    .progress-container {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 4px solid #3b82f6;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        border: 1px solid #e2e8f0;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f5f9;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #3b82f6 0%, #1e40af 100%);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(180deg, #1e40af 0%, #1e3a8a 100%);
    }
    
    /* Form styling */
    .stSelectbox > div > div {
        background-color: white;
        border: 2px solid #e2e8f0;
        border-radius: 8px;
    }
    
    .stSelectbox > div > div:focus-within {
        border-color: #3b82f6;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
    }
    
    .stNumberInput > div > div > input {
        background-color: white;
        border: 2px solid #e2e8f0;
        border-radius: 8px;
    }
    
    .stNumberInput > div > div > input:focus {
        border-color: #3b82f6;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main-header h1 {
            font-size: 2rem;
        }
        
        .main-header h3 {
            font-size: 1rem;
        }
        
        .section-header {
            font-size: 1.1rem;
            padding: 0.8rem 1rem;
        }
        
        .metric-card, .feature-card {
            padding: 1rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'model_results' not in st.session_state:
    st.session_state.model_results = None
if 'data_params' not in st.session_state:
    st.session_state.data_params = {'n_samples': DEFAULT_SAMPLE_SIZE, 'seed': DEFAULT_SEED}

# Reset any corrupted session state
if st.session_state.model_results is not None and not isinstance(st.session_state.model_results, dict):
    st.session_state.model_results = None


def main():
    """The main app function - handles the overall layout and flow."""
    
    # Header with clear instructions
    st.markdown(f"""
    <div class="main-header">
        <h1>{APP_TITLE}</h1>
        <h3>{APP_SUBTITLE}</h3>
        <div style="margin-top: 1rem; padding: 1rem; background: rgba(255,255,255,0.1); border-radius: 8px;">
            <h4 style="margin: 0 0 0.5rem 0;">What This Application Does:</h4>
            <p style="margin: 0; font-size: 1rem; line-height: 1.4;">
                <strong>1. Generate Data:</strong> Create synthetic patient records with realistic medical characteristics<br>
                <strong>2. Analyze Data:</strong> Explore correlations and patterns in the patient data<br>
                <strong>3. Train Model:</strong> Build a machine learning model to predict recovery time<br>
                <strong>4. Make Predictions:</strong> Input patient details to get personalized recovery estimates<br>
                <strong>5. Export Results:</strong> Download datasets and comprehensive model reports
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Quick status indicator
    if st.session_state.data is not None:
        st.markdown("""
        <div class="success-box">
            <h4>Dataset Loaded</h4>
            <p>Ready for analysis and modeling. Use the controls below to explore the data and train models.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="info-box">
            <h4>Get Started</h4>
            <p>Welcome to MediSense AI! Use the sidebar controls to generate synthetic clinical data and begin your analysis.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Sidebar controls
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0; border-bottom: 2px solid #e2e8f0; margin-bottom: 1rem;">
            <h2 style="margin: 0; color: #1e293b;">Control Panel</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Data generation controls
        st.markdown("### Data Generation")
        
        col1, col2 = st.columns(2)
        with col1:
            n_samples = st.slider(
                "Sample Size", 
                min_value=100, 
                max_value=2000, 
                value=st.session_state.data_params['n_samples'],
                step=50,
                help="Number of synthetic patient records to generate"
            )
        
        with col2:
            seed = st.number_input(
                "Random Seed", 
                min_value=0, 
                max_value=9999, 
                value=st.session_state.data_params['seed'],
                help="For reproducible results"
            )
        
        if st.button("Generate Data", type="primary", width='stretch'):
            with st.spinner("Generating synthetic clinical data..."):
                try:
                    st.session_state.data = generate_synthetic_data(n_samples, seed)
                    st.session_state.data_params = {'n_samples': n_samples, 'seed': seed}
                    st.session_state.model_results = None  # Reset model when new data
                    st.success("Data generated successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        
        st.markdown("---")
        
        # App settings
        st.markdown("### Settings")
        save_plots = st.toggle(
            "Save Plots to Disk", 
            value=False,
            help="Save generated plots to results/ folder"
        )
        
        st.markdown("---")
        
        # Quick actions
        st.markdown("### Quick Actions")
        
        if st.button("Auto-Generate & Train", type="secondary", width='stretch'):
            if st.session_state.data is None:
                with st.spinner("Auto-generating data..."):
                    st.session_state.data = generate_synthetic_data(n_samples, seed)
                    st.session_state.data_params = {'n_samples': n_samples, 'seed': seed}
            
            if st.session_state.data is not None:
                with st.spinner("Training model..."):
                    try:
                        model, results = train_model(st.session_state.data, random_state=seed)
                        st.session_state.model_results = results
                        st.success("Model trained successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
        
        # Status indicators
        st.markdown("---")
        st.markdown("### Status")
        
        if st.session_state.data is not None:
            st.success(f"Data: {len(st.session_state.data)} records")
        else:
            st.warning("No data loaded")
            
        if st.session_state.model_results is not None:
            st.success("Model trained")
        else:
            st.info("Model not trained")
    
    # Main content area
    if st.session_state.data is None:
        show_welcome_screen()
    else:
        show_main_content(save_plots)


def show_welcome_screen():
    """Shows the welcome screen when there's no data loaded yet."""
    
    st.markdown("""
    <div class="info-box">
        <h3>Welcome to MediSense AI</h3>
        <p>This application demonstrates machine learning for post-surgical recovery time prediction using synthetic clinical data.</p>
        <p><strong>To get started:</strong></p>
        <ol>
            <li>Use the sidebar to configure data generation parameters</li>
            <li>Click "Generate Data" to create synthetic patient records</li>
            <li>Explore the data with visualizations and analysis tools</li>
            <li>Train a machine learning model to predict recovery times</li>
            <li>Use the interactive prediction form to test individual cases</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)


def show_main_content(save_plots):
    """Shows the main content when we have data loaded."""
    
    # Data Overview Section
    st.markdown('<div class="section-header">Data Overview</div>', unsafe_allow_html=True)
    
    # Basic info with enhanced styling
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            "Total Records", 
            len(st.session_state.data),
            help="Number of synthetic patient records"
        )
    with col2:
        st.metric(
            "Features", 
            len(st.session_state.data.columns) - 1,  # Exclude target
            help="Number of input features for prediction"
        )
    with col3:
        avg_recovery = st.session_state.data['Recovery_Time'].mean()
        st.metric(
            "Avg Recovery Time", 
            f"{avg_recovery:.1f} days",
            help="Average recovery time across all patients"
        )
    with col4:
        st.metric(
            "Data Quality", 
            "100%",
            help="Percentage of complete records (no missing values)"
        )
    
    # Data summary
    with st.expander("View Data Summary", expanded=False):
        summary = get_data_summary(st.session_state.data)
        st.json(summary)
    
    st.markdown("---")
    
    # EDA & Visuals Section
    st.markdown('<div class="section-header">Exploratory Data Analysis</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        <div class="info-box">
            <h4>Data Visualization</h4>
            <p>Generate correlation heatmaps and feature distributions to understand your data patterns and relationships.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        if st.button("Generate EDA Plots", type="primary", width='stretch'):
            with st.spinner("Creating visualizations..."):
                try:
                    # Correlation heatmap
                    st.markdown("### Correlation Matrix")
                    fig_corr = create_correlation_heatmap(st.session_state.data)
                    st.pyplot(fig_corr)
                    st.caption("**Interpretation:** Darker colors indicate stronger relationships between features. Red = positive correlation, Blue = negative correlation.")
                    
                    if save_plots:
                        os.makedirs(RESULTS_DIR, exist_ok=True)
                        fig_corr.savefig(f"{RESULTS_DIR}/corr_heatmap.png", dpi=300, bbox_inches='tight')
                        st.success(f"Correlation plot saved to {RESULTS_DIR}/corr_heatmap.png")
                    
                    # Feature distributions
                    st.markdown("### Feature Distributions")
                    fig_dist = create_feature_distributions(st.session_state.data)
                    st.pyplot(fig_dist)
                    st.caption("**Interpretation:** These histograms show how each feature is distributed. Look for normal distributions, skewness, or unusual patterns.")
                    
                    if save_plots:
                        fig_dist.savefig(f"{RESULTS_DIR}/feature_distributions.png", dpi=300, bbox_inches='tight')
                        st.success(f"Distribution plots saved to {RESULTS_DIR}/feature_distributions.png")
                    
                except Exception as e:
                    st.error(f"Error creating plots: {str(e)}")
    
    st.markdown("---")
    
    # Modeling Section
    st.markdown('<div class="section-header">Machine Learning Model</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        <div class="info-box">
            <h4>Model Training</h4>
            <p>Train a Random Forest regression model to predict recovery time based on patient characteristics. The model will be evaluated using standard regression metrics.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        if st.button("Train Model", type="primary", width='stretch'):
            with st.spinner("Training machine learning model..."):
                try:
                    model, results = train_model(st.session_state.data, random_state=st.session_state.data_params['seed'])
                    st.session_state.model_results = results
                    st.success("Model trained successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    # Display model results if available
    if st.session_state.model_results is not None:
        display_model_results(save_plots)
    
    st.markdown("---")
    
    # Interactive Prediction Section
    st.markdown('<div class="section-header">Interactive Prediction Demo</div>', unsafe_allow_html=True)
    
    if st.session_state.model_results is not None:
        show_prediction_form()
    else:
        st.markdown("""
        <div class="warning-box">
            <h4>Model Required</h4>
            <p>Please train a model first using the "Train Model" button above to enable predictions.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Export Section
    st.markdown('<div class="section-header">Reproducibility & Export</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h4>Export Data</h4>
            <p>Download the current dataset as a CSV file for external analysis or backup.</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Download Dataset", width='stretch'):
            csv = st.session_state.data.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"medisense_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h4>Generate Report</h4>
            <p>Create a comprehensive model report with metrics, visualizations, and analysis.</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Generate Report", width='stretch'):
            if st.session_state.model_results is not None:
                with st.spinner("Generating comprehensive report..."):
                    try:
                        report = generate_model_report(
                            st.session_state.data,
                            st.session_state.model_results,
                            st.session_state.data_params
                        )
                        
                        st.download_button(
                            label="Download Report",
                            data=report,
                            file_name=f"medisense_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                            mime="text/markdown"
                        )
                        
                        if save_plots:
                            st.success("Report generated and plots saved to results/ folder")
                        else:
                            st.success("Report generated successfully")
                            
                    except Exception as e:
                        st.error(f"Error generating report: {str(e)}")
            else:
                st.warning("Please train a model first to generate a report")


def display_model_results(save_plots):
    """Shows the model results and metrics after training."""
    
    model_results = st.session_state.model_results
    
    # Model metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="margin: 0; color: #1e40af;">R² Score</h3>
            <h1 style="margin: 0.5rem 0; color: #1e293b;">{model_results['test_metrics']['r2']:.3f}</h1>
            <p style="margin: 0; color: #64748b;">Coefficient of determination</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="margin: 0; color: #1e40af;">MAE</h3>
            <h1 style="margin: 0.5rem 0; color: #1e293b;">{model_results['test_metrics']['mae']:.2f}</h1>
            <p style="margin: 0; color: #64748b;">Mean Absolute Error (days)</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="margin: 0; color: #1e40af;">RMSE</h3>
            <h1 style="margin: 0.5rem 0; color: #1e293b;">{model_results['test_metrics']['rmse']:.2f}</h1>
            <p style="margin: 0; color: #64748b;">Root Mean Square Error (days)</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Feature importance
    st.markdown("### Feature Importance")
    fig_importance = create_feature_importance_plot(model_results['feature_importance'])
    st.pyplot(fig_importance)
    
    if save_plots:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        fig_importance.savefig(f"{RESULTS_DIR}/feature_importance.png", dpi=300, bbox_inches='tight')
        st.success(f"Feature importance plot saved to {RESULTS_DIR}/feature_importance.png")
    
    # Predicted vs Actual
    st.markdown("### Model Performance Visualization")
    fig_performance = create_predicted_vs_actual_plot(
        model_results['y_test'], 
        model_results['y_test_pred']
    )
    st.pyplot(fig_performance)
    
    if save_plots:
        fig_performance.savefig(f"{RESULTS_DIR}/predicted_vs_actual.png", dpi=300, bbox_inches='tight')
        st.success(f"Performance plot saved to {RESULTS_DIR}/predicted_vs_actual.png")


def show_prediction_form():
    """Shows the form where you can input patient data and get predictions."""
    
    st.markdown("""
    <div class="info-box">
        <h4>Patient Prediction Form</h4>
        <p>Enter patient characteristics below to get a personalized recovery time prediction. The model will provide an estimate based on the trained algorithm.</p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input(
                "Age (years)", 
                min_value=18, 
                max_value=100, 
                value=45,
                help="Patient age in years"
            )
            
            bmi = st.number_input(
                "BMI (kg/m²)", 
                min_value=15.0, 
                max_value=50.0, 
                value=25.0,
                step=0.1,
                help="Body Mass Index"
            )
            
            heart_rate = st.number_input(
                "Resting Heart Rate (bpm)", 
                min_value=40, 
                max_value=120, 
                value=70,
                help="Resting heart rate in beats per minute"
            )
        
        with col2:
            pain_level = st.slider(
                "Pain Level (1-10)", 
                min_value=1, 
                max_value=10, 
                value=5,
                help="Self-reported pain level on a scale of 1-10"
            )
            
            mobility_score = st.slider(
                "Mobility Score (1-10)", 
                min_value=1, 
                max_value=10, 
                value=7,
                help="Mobility assessment score (1=very limited, 10=excellent)"
            )
        
        submitted = st.form_submit_button("Predict Recovery Time", type="primary", width='stretch')
        
        if submitted:
            # Create input DataFrame for prediction
            input_data = pd.DataFrame({
                'Age': [age],
                'BMI': [bmi], 
                'Heart_Rate': [heart_rate],
                'Pain_Level': [pain_level],
                'Mobility_Score': [mobility_score]
            })
            
            # Get the trained model
            model = st.session_state.model_results['model']
            
            # Make prediction
            prediction = model.predict(input_data)[0]
            
            # Display results
            st.markdown("---")
            
            # Main prediction display
            col1, col2, col3 = st.columns([1, 1, 1])
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h3 style="margin: 0; color: #1e40af;">Predicted Recovery Time</h3>
                    <h1 style="margin: 0.5rem 0; color: #1e293b;">{prediction:.1f} days</h1>
                    <p style="margin: 0; color: #64748b;">Estimated time to full recovery</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                # Get confidence statement
                mae = st.session_state.model_results['test_metrics']['mae']
                n_test = len(st.session_state.model_results['y_test'])
                
                if mae < 5:
                    confidence = "High"
                    confidence_color = "#10b981"
                elif mae < 10:
                    confidence = "Medium"
                    confidence_color = "#f59e0b"
                else:
                    confidence = "Low"
                    confidence_color = "#ef4444"
                
                st.markdown(f"""
                <div class="metric-card">
                    <h3 style="margin: 0; color: {confidence_color};">Model Confidence</h3>
                    <h1 style="margin: 0.5rem 0; color: #1e293b;">{confidence}</h1>
                    <p style="margin: 0; color: #64748b;">Based on test performance</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <h3 style="margin: 0; color: #1e40af;">Patient Profile</h3>
                    <p style="margin: 0; color: #64748b; font-size: 0.9rem;">
                        Age: {age} years<br>
                        BMI: {bmi:.1f}<br>
                        Heart Rate: {heart_rate} bpm<br>
                        Pain Level: {pain_level}/10<br>
                        Mobility: {mobility_score}/10
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            # Clinical interpretation
            st.markdown("### Clinical Interpretation")
            
            if prediction < 7:
                interpretation = "Excellent prognosis - patient likely to recover quickly with minimal complications."
                interpretation_color = "#10b981"
            elif prediction < 14:
                interpretation = "Good prognosis - standard recovery timeline expected with appropriate care."
                interpretation_color = "#3b82f6"
            elif prediction < 21:
                interpretation = "Moderate prognosis - may require extended monitoring and rehabilitation."
                interpretation_color = "#f59e0b"
            else:
                interpretation = "Complex case - may require specialized care and extended recovery planning."
                interpretation_color = "#ef4444"
            
            st.markdown(f"""
            <div class="info-box" style="border-left-color: {interpretation_color};">
                <h4>Clinical Assessment</h4>
                <p style="color: {interpretation_color}; font-weight: 600;">{interpretation}</p>
                <p><strong>Note:</strong> This prediction is based on synthetic data and should not be used for actual clinical decision-making.</p>
            </div>
            """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
