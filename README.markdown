# Company Bankruptcy Prediction

## Overview
This project predicts whether a company is likely to go bankrupt based on financial metrics using a Random Forest Classifier. It includes Exploratory Data Analysis (EDA) with 15 different analyses, data preprocessing, model training, and an interactive Streamlit application for visualizing results and making real-time predictions. The dataset (`data.csv`) contains numerical features such as ROA, Operating Profit Rate, and Debt Ratios, with a binary target variable `Bankrupt?` (0 or 1).

## Features
- **EDA**: 15 analyses including summary statistics, correlation analysis, visualizations (histograms, box plots, violin plots, pie charts), missing value analysis, outlier detection, and more.
- **Preprocessing**: Handles missing values, removes outliers using IQR, scales numerical features, and splits data into train/test sets.
- **Model**: Random Forest Classifier with accuracy ~0.95 and F1-score ~0.60 for the minority class (bankrupt).
- **Streamlit App**: Interactive interface for viewing EDA visualizations, inputting feature values, and predicting bankruptcy status with probabilities.

## Project Structure
```
bankruptcy_prediction/
├── data.csv                    # Input dataset
├── eda_preprocessing_linear_updated.py  # EDA and preprocessing script
├── model_training_linear.py    # Model training script
├── app_linear.py               # Streamlit app script
├── requirements.txt            # Python dependencies
├── model.pkl                   # Trained Random Forest model
├── scaler.pkl                  # StandardScaler for feature scaling
├── preprocessed_data.pkl       # Preprocessed train/test data and feature names
├── correlation_matrix.png      # Correlation heatmap
├── bankruptcy_distribution.png # Target variable count plot
├── bankruptcy_pie.png          # Target variable percentage pie chart
├── feature_distributions.png   # Feature histograms
├── boxplots.png                # Outlier detection box plots
├── pairplot.png                # Pairwise feature relationships
├── top_correlated_distributions.png  # Top correlated feature histograms
├── violin_plots.png            # Violin plots by bankruptcy status
├── README.md                   # Project documentation
```

## Setup Instructions
### Prerequisites
- Python 3.8+
- Git (optional, for Hugging Face deployment)
- Hugging Face account (for deployment)

### Install Dependencies
1. Clone the repository (if using Git):
   ```bash
   git clone https://github.com/<your-username>/bankruptcy_prediction.git
   cd bankruptcy_prediction
   ```
2. Install required Python packages:
   ```bash
   pip install -r requirements.txt
   ```
   Content of `requirements.txt`:
   ```
   streamlit==1.29.0
   pandas==2.2.2
   numpy==1.26.4
   scikit-learn==1.5.1
   matplotlib==3.9.2
   seaborn==0.13.2
   scipy==1.14.1
   ```

### Run Locally
1. **Perform EDA and Preprocessing**:
   - Ensure `data.csv` is in the project directory.
   - Run:
     ```bash
     python eda_preprocessing_linear_updated.py
     ```
   - Outputs: `preprocessed_data.pkl`, `scaler.pkl`, and visualization PNG files.

2. **Train the Model**:
   - Run:
     ```bash
     python model_training_linear.py
     ```
   - Outputs: `model.pkl` and evaluation metrics (accuracy, F1-score, etc.).

3. **Run the Streamlit App**:
   - Ensure `model.pkl`, `scaler.pkl`, `preprocessed_data.pkl`, and all PNG files are in the directory.
   - Run:
     ```bash
     streamlit run app_linear.py
     ```
   - Open `http://localhost:8501` in your browser to interact with the app.

## Deployment on Hugging Face Spaces
1. **Create a Space**:
   - Log in to [Hugging Face](https://huggingface.co/).
   - Go to **Spaces** > **New Space**.
   - Configure:
     - Name: e.g., `BankruptcyPredictionApp`
     - Framework: Streamlit
     - Visibility: Public (or Private with Pro account)
     - Hardware: Free tier (CPU)
     - Click **Create Space**.

2. **Upload Files**:
   - **Via Git**:
     - Clone the Space repository:
       ```bash
       git clone https://huggingface.co/spaces/<your-username>/<space-name>
       ```
     - Copy all files (`app_linear.py`, `model.pkl`, `scaler.pkl`, `preprocessed_data.pkl`, PNG files, `requirements.txt`) into the repository.
     - Commit and push:
       ```bash
       cd <space-name>
       git add .
       git commit -m "Add Streamlit app and dependencies"
       git push origin main
       ```
   - **Via Web Interface**:
     - Go to **Files and versions** tab in the Space.
     - Click **+ Add file** > **Upload files**.
     - Upload `app_linear.py`, `model.pkl`, `scaler.pkl`, `preprocessed_data.pkl`, PNG files, and `requirements.txt`.
     - Commit changes.

3. **Deploy and Test**:
   - Hugging Face will build the app after detecting `app_linear.py` and `requirements.txt`.
   - Monitor the **Logs** tab for build status.
   - Access the app at `https://<your-username>-<space-name>.hf.space`.
   - Test visualizations and predictions.

4. **Troubleshooting**:
   - Check logs for missing files or dependency issues.
   - Ensure all pickle and PNG files are in the root directory.
   - Re-run local scripts if pickle files are corrupted.

## Usage
- **EDA Section**: View 8 visualizations (correlation matrix, bankruptcy distribution, pie chart, feature distributions, box plots, pairplot, top correlated features, violin plots) to understand the dataset.
- **Prediction Section**: Enter values for the top 10 features to predict bankruptcy status and view the probability.
- **Conclusion**: Summary of key findings and future improvements.

## Future Improvements
- Handle class imbalance using SMOTE or other techniques.
- Experiment with other models (e.g., XGBoost, SVM).
- Add dynamic EDA in the Streamlit app using `data.csv`.
- Include all features in the prediction input form.

## License
MIT License

## Contact
For questions or contributions, contact <your-email> or open an issue on the repository.