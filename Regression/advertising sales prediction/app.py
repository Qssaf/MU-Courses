import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Step 1: Configure the Streamlit Page
st.set_page_config(page_title="Advertising Sales Prediction", layout="wide")
st.title("Advertising Sales Prediction - Regression Model")

# Step 2: Persistence - Load the trained model and preprocessor
try:
    model = joblib.load("model.pkl")
    preprocessor = joblib.load("preprocessor.pkl")
    dataset = pd.read_csv("Advertising.csv")
    results_df = pd.read_csv("actual_vs_predicted.csv")
    
    # Clean dataset if it has the unnamed index column
    if "Unnamed: 0" in dataset.columns:
        dataset = dataset.drop(columns=["Unnamed: 0"])
except Exception as e:
    st.error(f"Error loading files: {e}")
    st.stop()

# Define features based on the dataset
FEATURES = ["TV", "radio", "newspaper"]

# Step 3: Define Tabs for better organization
tab1, tab2, tab3 = st.tabs(["Real Dataset", "Model Result", "New Prediction"])

# Section: Dataset Exploration
with tab1:
    st.header("Original Advertising Dataset")
    st.write(f"Total records: {len(dataset)}")
    st.write(f"Features: {len(FEATURES)}")
    st.dataframe(dataset)

# Section: Model Evaluation
with tab2:
    st.header("Model Performance on Test Set")

    # Calculate Regression Metrics
    r2 = r2_score(results_df["Actual"], results_df["Predicted"])
    mae = mean_absolute_error(results_df["Actual"], results_df["Predicted"])
    mse = mean_squared_error(results_df["Actual"], results_df["Predicted"])
    rmse = np.sqrt(mse)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("R-squared (R2)", f"{r2:.3f}")
    col2.metric("MAE", f"{mae:.2f}")
    col3.metric("RMSE", f"{rmse:.2f}")
    col4.metric("Test Samples", len(results_df))

    st.subheader("Visualizing Predictions")
    
    col_plot, col_data = st.columns(2)

    with col_plot:
        st.write("Actual vs Predicted Sales")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(data=results_df, x="Actual", y="Predicted", alpha=0.7, ax=ax)
        
        # Add diagonal reference line
        max_val = max(results_df["Actual"].max(), results_df["Predicted"].max())
        min_val = min(results_df["Actual"].min(), results_df["Predicted"].min())
        ax.plot([min_val, max_val], [min_val, max_val], color='red', lw=2, linestyle='--')
        
        ax.set_xlabel("Actual Sales")
        ax.set_ylabel("Predicted Sales")
        st.pyplot(fig)

    with col_data:
        st.write("Detailed Comparison")
        st.dataframe(results_df)

# Section: New Instance Inference
with tab3:
    st.header("Predict Sales for New Advertising Budget")
    st.write("Enter advertisement spending manually to estimate sales.")

    # Example values for guidance
    example_values = {
        "TV": 150.0,
        "radio": 25.0,
        "newspaper": 30.0
    }

    user_input = {}
    
    col_in1, col_in2, col_in3 = st.columns(3)
    
    with col_in1:
        user_input["TV"] = st.number_input("TV Budget ($)", min_value=0.0, value=example_values["TV"], step=1.0)
    with col_in2:
        user_input["radio"] = st.number_input("Radio Budget ($)", min_value=0.0, value=example_values["radio"], step=1.0)
    with col_in3:
        user_input["newspaper"] = st.number_input("Newspaper Budget ($)", min_value=0.0, value=example_values["newspaper"], step=1.0)

    # Execution of the prediction when the button is pressed
    if st.button("Predict Sales"):
        # 1. Convert user inputs into a DataFrame
        input_df = pd.DataFrame([user_input])
        
        # 2. Scale/Transform the new data using the pre-fitted transformer
        input_processed = preprocessor.transform(input_df)

        # 3. Use the model to predict
        prediction = model.predict(input_processed)[0]

        st.subheader("Prediction Result")
        st.metric("Estimated Sales", f"{prediction:.2f} units")

        # Insight: Compare to average sales in the dataset
        avg_sales = dataset["sales"].mean()
        diff = prediction - avg_sales
        
        if diff > 0:
            st.success(f"This prediction is **{abs(diff):.2f} units above** the dataset average ({avg_sales:.2f}).")
        else:
            st.warning(f"This prediction is **{abs(diff):.2f} units below** the dataset average ({avg_sales:.2f}).")

        with st.expander("Show entered values"):
            st.dataframe(input_df)
