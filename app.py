import streamlit as st
import pickle
import numpy as np
import pandas as pd
import os

# --- Configuration ---
MODEL_FILE = 'N.pkl'
DATA_FILE = 'student_scores (1).csv'

# Set page configuration
st.set_page_config(
    page_title="Student Score Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Function to load the model
@st.cache_resource
def load_model(file_path):
    """Loads the pickled machine learning model."""
    if not os.path.exists(file_path):
        st.error(f"Error: Model file '{file_path}' not found. Please ensure it is uploaded.")
        return None
    try:
        with open(file_path, 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.info("The model might have been trained with a different version of scikit-learn/Python. Please check your environment.")
        return None

# Load the model
model = load_model(MODEL_FILE)

# --- App UI ---

st.title("🎓 Student Performance Score Predictor")
st.markdown("A simple regression model deployment using Streamlit and a pre-trained `LinearRegression` model.")

if model is None:
    st.stop()

# Input Section
st.header("Input Features")
st.markdown("Please enter the values for the student's performance metrics.")

# Use st.columns for a clean layout
col1, col2, col3 = st.columns(3)

with col1:
    hours = st.number_input(
        "Hours Studied (e.g., 5.0)",
        min_value=0.0,
        max_value=100.0,
        value=5.0,
        step=0.5,
        key="hours_input",
        format="%.1f"
    )

with col2:
    attendance = st.slider(
        "Attendance (%)",
        min_value=0,
        max_value=100,
        value=85,
        step=1,
        key="attendance_input"
    )

with col3:
    assignments = st.number_input(
        "Assignments Submitted (out of 10)",
        min_value=0,
        max_value=10,
        value=8,
        step=1,
        key="assignments_input"
    )

# Prediction Button
if st.button("Predict Score", type="primary"):
    # Prepare the input data for the model
    # The model expects a 2D array: [[Hours, Attendance, Assignments]]
    try:
        input_data = np.array([[hours, attendance, assignments]])
        
        # Make prediction
        predicted_score = model.predict(input_data)[0]
        
        # Ensure the score is within a realistic range [0, 100]
        final_score = np.clip(predicted_score, 0, 100)
        
        # Display results
        st.success("--- Prediction Result ---")
        st.balloons()
        
        st.metric(
            label="Predicted Final Score (out of 100)", 
            value=f"{final_score:.2f}",
            delta_color="off"
        )
        
        st.info(f"The model estimates a score of **{final_score:.2f}** based on your inputs.")
        
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.warning("Please ensure the input format matches what the model was trained on (e.g., correct feature order and data types).")


# --- Data Visualization (Optional: for context) ---
st.sidebar.header("Data Overview")

@st.cache_data
def load_data(file_path):
    """Loads the original training data."""
    if not os.path.exists(file_path):
        return None
    try:
        return pd.read_csv(file_path)
    except Exception:
        return None

data_df = load_data(DATA_FILE)

if data_df is not None:
    st.sidebar.subheader("Original Training Data Snippet")
    st.sidebar.dataframe(data_df.head(), use_container_width=True)
    
    # Simple chart to show distribution of scores
    st.sidebar.subheader("Score Distribution")
    st.sidebar.bar_chart(data_df['Score'])

    st.sidebar.caption("The model was trained on data similar to this.")
else:
    st.sidebar.info(f"Could not load the original data file ('{DATA_FILE}') for context.")

st.markdown("---")
st.caption("Model Type: Linear Regression | Features: Hours_Studied, Attendance, Assignments_Submitted")
