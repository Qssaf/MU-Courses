import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Step 1: Configure the Streamlit Page
st.set_page_config(page_title="Cars Price Prediction", layout="wide")
st.title("Cars Price Prediction - Regression Model")

# Step 2: Persistence - Load the trained model and preprocessor
try:
    model = joblib.load("model.pkl")
    preprocessor = joblib.load("preprocessor.pkl")
    dataset = pd.read_csv("car - prediction.csv")
    results_df = pd.read_csv("actual_vs_predicted.csv")
except Exception as e:
    st.error(f"Error loading files: {e}")
    st.info("Make sure 'model.pkl', 'preprocessor.pkl', 'car - prediction.csv', and 'actual_vs_predicted.csv' are in the project folder.")
    st.stop()

# Define feature columns
FEATURES = ["name", "year", "km_driven", "fuel", "seller_type", "transmission", "owner"]

# Step 3: Define Tabs for better organization
tab1, tab2, tab3 = st.tabs(["Real Dataset", "Model Result", "New Prediction"])

# Section: Dataset Exploration
with tab1:
    st.header("Original Cars Dataset")
    st.write(f"Total records: {len(dataset)}")
    st.write(f"Features used: {len(FEATURES)}")
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
    col2.metric("MAE", f"₹{mae:,.0f}")
    col3.metric("RMSE", f"₹{rmse:,.0f}")
    col4.metric("Test Samples", len(results_df))

    st.subheader("Visualizing Predictions")
    
    col_plot, col_data = st.columns(2)

    with col_plot:
        st.write("Actual vs Predicted Price")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(data=results_df, x="Actual", y="Predicted", alpha=0.5, ax=ax)
        
        # Add diagonal reference line
        max_val = max(results_df["Actual"].max(), results_df["Predicted"].max())
        min_val = min(results_df["Actual"].min(), results_df["Predicted"].min())
        ax.plot([min_val, max_val], [min_val, max_val], color='red', lw=2, linestyle='--')
        
        ax.set_xlabel("Actual Price")
        ax.set_ylabel("Predicted Price")
        st.pyplot(fig)

    with col_data:
        st.write("Detailed Comparison")
        st.dataframe(results_df)

# Section: New Instance Inference
with tab3:
    st.header("Predict Selling Price for a Car")
    st.write("Enter the details of the car to estimate its market value.")

    # Prepare unique options for selectboxes from the dataset
    unique_names = sorted(dataset["name"].unique())
    unique_fuels = sorted(dataset["fuel"].unique())
    unique_sellers = sorted(dataset["seller_type"].unique())
    unique_transmissions = sorted(dataset["transmission"].unique())
    unique_owners = sorted(dataset["owner"].unique())

    # Create UI for input
    col_in1, col_in2 = st.columns(2)
    
    user_input = {}
    
    with col_in1:
        user_input["name"] = st.selectbox("Car Model Name", unique_names)
        user_input["year"] = st.number_input("Manufacture Year", min_value=1990, max_value=2024, value=2015)
        user_input["km_driven"] = st.number_input("Kilometers Driven", min_value=0, value=50000, step=1000)
        user_input["fuel"] = st.selectbox("Fuel Type", unique_fuels)
        
    with col_in2:
        user_input["seller_type"] = st.selectbox("Seller Type", unique_sellers)
        user_input["transmission"] = st.selectbox("Transmission", unique_transmissions)
        user_input["owner"] = st.selectbox("Owner Type", unique_owners)

    # Execution of the prediction when the button is pressed
    if st.button("Predict Price"):
        # 1. Convert user inputs into a DataFrame
        input_df = pd.DataFrame([user_input])
        
        # 2. Scale/Transform the new data using the pre-fitted transformer
        # Note: 'handle_unknown' is crucial for car names
        try:
            input_processed = preprocessor.transform(input_df)
            
            # 3. Use the model to predict
            prediction = model.predict(input_processed)[0]

            st.subheader("Prediction Result")
            st.success(f"Estimated Market Value: **₹{prediction:,.2f}**")

            # Insight: Compare to average price in the dataset
            avg_price = dataset["selling_price"].mean()
            diff = prediction - avg_price
            
            if diff > 0:
                st.info(f"This car is estimated to be **₹{abs(diff):,.2f} above** the dataset average.")
            else:
                st.warning(f"This car is estimated to be **₹{abs(diff):,.2f} below** the dataset average.")

        except Exception as e:
            st.error(f"Error during prediction: {e}")
            st.info("This might happen if the name/category wasn't in the training set and 'handle_unknown' wasn't set to 'ignore'.")

        with st.expander("Show entered data"):
            st.dataframe(input_df)
