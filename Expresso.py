# Import necessary libraries
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# Set page config
st.set_page_config(
    page_title="Expresso Churn Prediction",
    page_icon="📱",
    layout="wide"
)

# Title
st.title("Expresso Customer Churn Prediction")

# Sidebar
st.sidebar.header("Upload Data")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

def process_data(df):
    # Make a copy
    df = df.copy()
    
    # Handle missing values
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    categorical_columns = df.select_dtypes(include=['object']).columns
    
    # Fill numeric columns with median
    for col in numeric_columns:
        df[numeric_columns] = df[numeric_columns].apply(lambda col: col.fillna(col.median()))
    
    # Fill categorical columns with mode
    for col in categorical_columns:
        df[categorical_columns] = df[categorical_columns].apply(lambda col: col.fillna(col.mode()[0]))
    
    # Drop duplicates
    df.drop_duplicates(inplace=True)
    
    # Handle outliers
    def remove_outliers(df, column):
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
        return df
    
    for col in numeric_columns:
        df = remove_outliers(df, col)
    
    # Encode categorical features
    label_encoders = {}
    for col in categorical_columns:
        label_encoders[col] = LabelEncoder()
        df[col] = label_encoders[col].fit_transform(df[col])
    
    return df, label_encoders

def train_model(df):
    x = df.drop('CHURN', axis=1)
    y = df['CHURN']
    
    # Split data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(x_train)
    X_test_scaled = scaler.transform(x_test)
    
    # Train model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Get model performance
    y_pred = model.predict(X_test_scaled)
    report = classification_report(y_test, y_pred)
    
    return model, scaler, x.columns.tolist(), report

if uploaded_file is not None:
    # Load data
    df = pd.read_csv(uploaded_file)
    
    # Show raw data
    st.subheader("Raw Data Preview")
    st.write(df.head())
    
    # Show data info
    st.subheader("Dataset Information")
    col1, col2 = st.columns(2)
    with col1:
        st.write("Dataset Shape:", df.shape)
    with col2:
        st.write("Missing Values:", df.isnull().sum().sum())
    
    # Process and train button
    if st.button("Process Data and Train Model"):
        with st.spinner("Processing data and training model..."):
            # Process data
            processed_df, label_encoders = process_data(df)
            
            # Train model
            model, scaler, feature_names, report = train_model(processed_df)
            
            # Save models and encoders
            joblib.dump(model, 'model.joblib')
            joblib.dump(scaler, 'scaler.joblib')
            joblib.dump(label_encoders, 'label_encoders.joblib')
            joblib.dump(feature_names, 'feature_names.joblib')
            
            # Show results
            st.success("Model trained successfully!")
            st.subheader("Model Performance Report")
            st.text(report)
            
            # Download buttons
            col1, col2, col3 = st.columns(3)
            with col1:
                with open('model.joblib', 'rb') as f:
                    st.download_button(
                        label="Download Model",
                        data=f,
                        file_name='model.joblib',
                        mime='application/octet-stream'
                    )
            with col2:
                with open('scaler.joblib', 'rb') as f:
                    st.download_button(
                        label="Download Scaler",
                        data=f,
                        file_name='scaler.joblib',
                        mime='application/octet-stream'
                    )
            with col3:
                with open('label_encoders.joblib', 'rb') as f:
                    st.download_button(
                        label="Download Encoders",
                        data=f,
                        file_name='label_encoders.joblib',
                        mime='application/octet-stream'
                    )
else:
    st.info("Please upload a CSV file to begin.")