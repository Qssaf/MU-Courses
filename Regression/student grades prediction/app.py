import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Step 1: Configure the Streamlit Page (Layout and Title)
st.set_page_config(page_title="Student Math Score Prediction", layout="wide")
st.title("Student Math Score Prediction - Linear Regression model")
st.success("**🎨 Style Note:** You can customize this application's look and feel! Experiment with different layouts, colors, and Streamlit components to make it your own. Also in the Hackathon!")

# Step 2: Load the persisted models and data
# Note: Ensure these files exist in the same directory as app.py
try:
    model = joblib.load("model.pkl")
    preprocessor = joblib.load("preprocessor.pkl")
    dataset = pd.read_csv("student_grades_prediction.csv")
    results_df = pd.read_csv("actual_vs_predicted.csv")
except Exception as e:
    st.error(f"Error loading files: {e}")
    st.stop()

FEATURES = [
    "gender", "race_ethnicity", "parental_level_of_education", 
    "lunch", "test_preparation_course", "reading_score", "writing_score"
]

# Step 3: Organize content into Tabs
tab0, tab1, tab2, tab3 = st.tabs(["Project Workflow", "Real Dataset", "Model Result", "New Prediction"])

# Section: Project Workflow (Methodology)
with tab0:
    st.header("Machine Learning Workflow")
    st.write("This project follows a systematic approach to predict student math scores using Linear Regression.")
    
    st.subheader("Data Analysis & Preprocessing")
    st.markdown("""
    - **Data Loading**: Collected student performance data containing demographic and academic features.
    - **Feature Engineering**: Identified categorical features (gender, race, education, etc.) and numerical features (reading and writing scores).
    - **Preprocessing**: 
        - Applied `OneHotEncoder` to convert categorical variables into numerical format.
        - Applied `StandardScaler` to normalize numerical scores.
    - **Data Splitting**: Divided the processed data into an 80% training set and a 20% test set.
    """)
    
    st.subheader("Model Selection & Training")
    st.markdown("""
    - **Algorithm**: Selected **Linear Regression** for its transparency and effectiveness in predicting continuous numerical outcomes like exam scores.
    - **Training**: Fitted the model on the training data to learn the relationships between demographics, study habits, and math performance.
    """)
    
    st.subheader("Evaluation & Persistence")
    st.markdown("""
    - **Evaluation**: Validated the model using performance metrics:
        - **R-squared (R2)**: Measures how well the model explains the variance in scores.
        - **Mean Absolute Error (MAE)**: Measures the average magnitude of errors in predictions.
        - **Mean Squared Error (MSE)**: Penalizes larger errors more heavily.
    - **Persistence**: Saved the trained model (`model.pkl`) and the preprocessor (`preprocessor.pkl`) for real-time predictions.
    """)

    st.subheader("Deployment & Interactive UI")
    st.markdown("""
    **Setup & Infrastructure:**
    1. **Environment Setup**: Ensure `streamlit`, `pandas`, `joblib`, and `scikit-learn` are installed.
    2. **Model Loading**: Load the `.pkl` files created during training.
    3. **UI Layout**: Use `st.tabs` to separate data exploration, evaluation, and inference.

    **Application Features:**
    1. **Data Explorer**: View raw student records.
    2. **Performance Dashboard**: Visualize prediction accuracy using scatter plots and metrics.
    3. **Prediction Tool**: Enter student details to estimate their math score instantly.
    """)
    
    st.info("💡 **Streamlit Requirement:** This app requires `model.pkl`, `preprocessor.pkl`, `student_grades_prediction.csv`, and `actual_vs_predicted.csv` in the same directory.")

# Section: Real Dataset (Exploration)
with tab1:
    st.header("Original Student Grades Dataset")
    st.write(f"Total records: {len(dataset)}")
    st.write(f"Total Features: {len(FEATURES)}")
    st.dataframe(dataset)

# Section: Model Result (Evaluation)
with tab2:
    st.header("Model Performance on Test Set")
    
    r2 = r2_score(results_df["Actual"], results_df["Predicted"])
    mae = mean_absolute_error(results_df["Actual"], results_df["Predicted"])
    mse = mean_squared_error(results_df["Actual"], results_df["Predicted"])
    rmse = np.sqrt(mse)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("R-squared (R2)", f"{r2:.3f}")
    col2.metric("Mean Absolute Error", f"{mae:.2f}")
    col3.metric("Root Mean Squared Error", f"{rmse:.2f}")
    col4.metric("Total Test Samples", len(results_df))

    st.subheader("Actual vs Predicted Scores")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=results_df, x="Actual", y="Predicted", alpha=0.6, ax=ax)
    # Add a diagonal line for reference
    min_val = min(results_df["Actual"].min(), results_df["Predicted"].min())
    max_val = max(results_df["Actual"].max(), results_df["Predicted"].max())
    ax.plot([min_val, max_val], [min_val, max_val], '--r', lw=2)
    
    ax.set_xlabel("Actual Math Score")
    ax.set_ylabel("Predicted Math Score")
    ax.set_title("Actual vs Predicted Math Scores")
    st.pyplot(fig)
    
    with st.expander("Show detailed results"):
        st.dataframe(results_df)

# Section: New Prediction (Inference)
with tab3:
    st.header("Predict Math Score for a New Student")
    st.write("Enter student details manually to get a prediction.")

    # Define options for select boxes
    gender_options = ["female", "male"]
    race_options = ["group A", "group B", "group C", "group D", "group E"]
    education_options = [
        "some high school", "high school", "some college", 
        "associate's degree", "bachelor's degree", "master's degree"
    ]
    lunch_options = ["standard", "free/reduced"]
    prep_options = ["none", "completed"]

    col_input1, col_input2 = st.columns(2)

    with col_input1:
        gender = st.selectbox("Gender", gender_options)
        race = st.selectbox("Race/Ethnicity", race_options)
        education = st.selectbox("Parental Level of Education", education_options)
        lunch = st.selectbox("Lunch Type", lunch_options)

    with col_input2:
        test_prep = st.selectbox("Test Preparation Course", prep_options)
        reading_score = st.number_input("Reading Score (0-100)", min_value=0, max_value=100, value=70)
        writing_score = st.number_input("Writing Score (0-100)", min_value=0, max_value=100, value=70)

    # Process the Prediction when the user clicks the button
    if st.button("Predict"):
        user_input = {
            "gender": gender,
            "race_ethnicity": race,
            "parental_level_of_education": education,
            "lunch": lunch,
            "test_preparation_course": test_prep,
            "reading_score": reading_score,
            "writing_score": writing_score
        }

        # Convert input to DataFrame for transformer
        student_df = pd.DataFrame([user_input])
        
        # Apply the preprocessor (scaling and encoding)
        student_processed = preprocessor.transform(student_df)
        
        # Get result from model
        prediction = model.predict(student_processed)[0]

        st.subheader("Prediction Result")
        st.metric("Predicted Math Score", f"{prediction:.2f}")
        
        # Why compare to average? 
        # 1. Context: A raw score like '72' needs a yardstick to see if it's high or low.
        # 2. Actionability: Helps teachers identify 'at-risk' students who fall below the norm.
        # 3. Hackathon Extra: Shows judges you can interpret data, not just predict it.
        avg_math = dataset["math_score"].mean()
        diff = prediction - avg_math
        if diff > 0:
            st.success(f"This predicted score is **{abs(diff):.2f} points above average** ({avg_math:.2f})")
        else:
            st.warning(f"This predicted score is **{abs(diff):.2f} points below average** ({avg_math:.2f})")

        with st.expander("Show entered values"):
            st.dataframe(student_df)
