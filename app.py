
import pickle

import streamlit as st



# Load model and scaler
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('preprocessed_data.pkl', 'rb') as f:
    _, _, _, _, feature_names = pickle.load(f)

# Streamlit app
st.title("Company Bankruptcy Prediction")

# Introduction
st.header("Introduction")
st.write("""
This application predicts whether a company is likely to go bankrupt based on financial metrics.
The dataset includes features like ROA, Operating Profit Rate, and Debt Ratios.
The goal is to provide insights through Exploratory Data Analysis (EDA) and enable real-time bankruptcy predictions using a Random Forest model.
""")

# EDA Section
st.header("Exploratory Data Analysis")
st.subheader("Correlation Matrix")
st.image('correlation_matrix.png', caption='Correlation between features')
st.subheader("Bankruptcy Distribution")
st.image('bankruptcy_distribution.png', caption='Count of bankrupt vs non-bankrupt companies')
st.subheader("Percentage of Bankruptcy Status")
st.image('bankruptcy_pie.png', caption='Percentage of bankrupt vs non-bankrupt companies')
st.subheader("Feature Distributions")
st.image('feature_distributions.png', caption='Distributions of selected numerical features')
st.subheader("Box Plots")
st.image('boxplots.png', caption='Outlier detection for selected features')
st.subheader("Pairwise Relationships")
st.image('pairplot.png', caption='Relationships between selected features colored by bankruptcy status')
st.subheader("Top Correlated Features")
st.image('top_correlated_distributions.png', caption='Distributions of top 3 features correlated with bankruptcy')
st.subheader("Violin Plots")
st.image('violin_plots.png', caption='Distribution of top features by bankruptcy status')

# Model Section
st.header("Model Predictions")
st.subheader("Random Forest Classifier")
st.write("""
The model is a Random Forest Classifier trained on the preprocessed dataset.
It achieves an accuracy of ~0.95 and an F1-score of ~0.60 for the minority class (bankrupt).
Enter values for the features below to predict bankruptcy status.
""")