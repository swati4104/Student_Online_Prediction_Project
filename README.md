# Student_Online_Prediction_Project

Below is My Live Application Link

--> https://studentonlinepredictionproject-rd3kk4cfd8jop7y7axsqfv.streamlit.app/


🎓 Student Performance Prediction Dashboard

Project Overview

This project is a simple, yet effective, machine learning application deployed using Streamlit to predict a student's final score based on key performance indicators (KPIs). The prediction is powered by a pre-trained Linear Regression model packaged as N.pkl. This tool provides educators, students, or administrators with immediate insights into potential academic outcomes based on metrics like study hours, attendance, and assignment completion.

Features

Interactive Prediction: Users can adjust three key input features (Hours Studied, Attendance, Assignments Submitted) using sliders and instantly see the predicted score.

Real-time Feedback: The application provides a visual score (out of 100) and conversational feedback categorized by performance level (Excellent, Good, Lower Score).

Model Transparency: Displays the underlying model coefficients and the intercept, showing the weight and influence of each input feature on the final score.

Responsive UI: Built with Streamlit for a clean, user-friendly, and responsive web interface.

Technical Details: Model and Data

Model (N.pkl)

Type: Scikit-learn Linear Regression model.

Purpose: Trained to find a linear relationship between input features and the final score.

Training Data: The model was trained on historical student data (referenced as student_scores (1).csv), which includes the following features:

Feature

Description

Hours_Studied

Average weekly hours spent studying.

Attendance

Course attendance percentage (%).

Assignments_Submitted

Total number of assignments completed.

Target Variable

Score (Final test score, 0-100).

Installation and Setup

To run this application locally, ensure you have Python installed and follow the steps below.

1. Clone the repository (or set up the files)

Ensure the following files are in the same directory:

app.py (The Streamlit application code)

N.pkl (The pickled Linear Regression model)

requirements.txt (List of dependencies)

2. Install Dependencies

Install all required Python packages using the provided requirements.txt file:

pip install -r requirements.txt


3. Run the Application

Execute the Streamlit application from your terminal:

streamlit run app.py


This command will start the application, and it will automatically open in your default web browser (usually at http://localhost:8501).

Dependencies

The project relies on the following Python libraries, as listed in requirements.txt:

streamlit: For creating the interactive web application.

pandas: For data handling and structuring input features.

numpy: For numerical operations.

scikit-learn: For loading and using the pickled machine learning model.

pickle: For deserializing the saved model object.
