import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Streamlit app configuration
st.set_page_config(page_title="Bankruptcy Prediction App", layout="wide")
st.title("Company Bankruptcy Prediction Dashboard")

# Load model, scaler, and feature names
with open('src/model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('src/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('src/preprocessed_data.pkl', 'rb') as f:
    _, _, _, _, feature_names = pickle.load(f)

# Sidebar navigation
st.sidebar.header("Navigation")
page = st.sidebar.selectbox("Choose a section", ["Home", "EDA", "Predict", "About"])

# Home section
if page == "Home":
    st.header("Welcome to the Bankruptcy Prediction App")
    st.write("""
    This application uses a Random Forest model to predict whether a company is likely to go bankrupt based on financial metrics.
    Explore the dataset through visualizations, make real-time predictions, or learn about the project.
    The dataset is from Kaggle's Company Bankruptcy Prediction collection, featuring financial ratios and a binary `Bankrupt?` label.
    Use the sidebar to navigate to different sections.
    """)

# EDA section
elif page == "EDA":
    st.header("Exploratory Data Analysis")
    st.write("Visualizations and insights from the company bankruptcy dataset.")

    # First row: Bankruptcy distribution and pie chart
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Bankruptcy Distribution")
        st.image('src/bankruptcy_distribution.png', caption='Count of Bankrupt vs Non-Bankrupt Companies')
    with col2:
        st.subheader("Bankruptcy Percentage")
        st.image('src/bankruptcy_pie.png', caption='Percentage Breakdown')

    # Second row: Correlation heatmap and feature distributions
    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Correlation Heatmap")
        st.image('src/correlation_matrix.png', caption='Feature Correlations')
    with col4:
        st.subheader("Feature Distributions")
        st.image('src/feature_distributions.png', caption='Distributions of Top Numerical Features')

    # Third row: Box plots and violin plots
    col5, col6 = st.columns(2)
    with col5:
        st.subheader("Box Plots")
        st.image('src/boxplots.png', caption='Outlier Detection for Top Features')
    with col6:
        st.subheader("Violin Plots")
        st.image('src/violin_plots.png', caption='Distribution by Bankruptcy Status')

    # Fourth row: Pairplot and top correlated features
    col7, col8 = st.columns(2)
    with col7:
        st.subheader("Pairwise Relationships")
        st.image('src/pairplot.png', caption='Relationships Between Top Features')
    with col8:
        st.subheader("Top Correlated Features")
        st.image('src/top_correlated_distributions.png', caption='Distributions of Top Correlated Features')

# Predict section
elif page == "Predict":
    st.header("Make a Bankruptcy Prediction")
    st.write("Enter feature values to predict if a company is likely to go bankrupt.")
    
    st.subheader("Model Details")
    st.write("""
    The Random Forest Classifier is trained with 100 trees and achieves an accuracy of ~0.95 and an F1-score of ~0.60 for the bankrupt class.
    Outlier removal and feature scaling enhance performance.
    Input the top 10 features below using sliders for prediction.
    Note: Features are scaled; use values between -5 and 5 for realistic inputs. Other features are set to default values (0 for scaled data).
    """)

    # Input sliders for top 10 features
    input_values = {}
    for feature in feature_names[:10]:
        input_values[feature] = st.slider(f"{feature}", -5.0, 5.0, 0.0)

    if st.button("Run Prediction"):
        # Create a DataFrame with all features, initialized to 0
        full_input = {feature: 0.0 for feature in feature_names}
        # Update with user-provided values for top 10 features
        full_input.update(input_values)
        # Convert to DataFrame with correct feature order
        input_df = pd.DataFrame([full_input], columns=feature_names)
        # Scale the input
        input_scaled = scaler.transform(input_df)
        # Make prediction
        pred = model.predict(input_scaled)[0]
        # Get probability
        prob_array = model.predict_proba(input_scaled)[0]
        # Handle single-class probability case
        prob = prob_array[1] if len(prob_array) > 1 else (0.0 if pred == 0 else 1.0)
        
        st.subheader("Result")
        st.write(f"**Prediction**: {'Bankrupt' if pred == 1 else 'Non-Bankrupt'}")
        st.write(f"**Bankruptcy Probability**: {prob:.2%}")

# About section
elif page == "About":
    st.header("About the Project")
    st.write("""
    **Objective**: Predict company bankruptcy using financial metrics with machine learning.
    **Dataset**: ~6,819 companies with 95 numerical features (e.g., ROA, Debt Ratios) and a binary `Bankrupt?` label, sourced from Kaggle.
    **Methodology**:
    - Conducted 15 EDA analyses including summary statistics, correlation analysis, outlier detection, and visualizations using Matplotlib/Seaborn.
    - Preprocessed data with median imputation for missing values, IQR-based outlier removal, and StandardScaler for feature scaling.
    - Trained a Random Forest Classifier with 100 trees, achieving high accuracy and moderate F1-score for the minority class.
    - Developed an interactive Streamlit app for data exploration and real-time predictions.
    **Key Findings**:
    - Bankruptcy is rare (~3.2% of companies), indicating an imbalanced dataset.
    - Features like Net Income to Total Assets and Debt Ratios are highly correlated with bankruptcy.
    - Outlier removal improves model robustness, but class imbalance affects F1-score.
    - The model is suitable for decision-making but could benefit from SMOTE or ensemble methods.
    **Future Work**:
    - Implement SMOTE or other oversampling techniques.
    - Test alternative models like XGBoost or LightGBM.
    - Add dynamic EDA capabilities to the app.
    """)
