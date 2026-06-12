# Import necessary libraries for UI, data handling, and machine learning evaluation
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

st.set_page_config(page_title="Car KNN Demo", layout="wide")
st.markdown("""
    <style>
    /* Main headers color */
    h1, h2, h3 {
        color: #004e98 !important;
    }
    /* Metrics customization */
    [data-testid="stMetricValue"] {
        color: #004e98;
    }
    /* Button styling */
    div.stButton > button {
        background-color: #004e98;
        color: white;
        border-radius: 8px;
        border: none;
    }
    div.stButton > button:hover {
        background-color: #003366;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)
st.title("Car Prediction")
if "dataframe" not in st.session_state:
        st.session_state.dataframe = pd.DataFrame()



# Step 1: Persistence - Load the trained model and preprocessor
knn_model = joblib.load("knn_model.pkl")
preprocessor = joblib.load("preprocessor.pkl")
if "dataframe" not in st.session_state:
    st.session_state.dataframe = pd.DataFrame()

# Step 2: Load the data
# Note: 'car_actual_vs_predicted_knn.csv' have columns named 'Actual' and 'Predicted'
dataset = pd.read_csv("car.csv")
results_df= pd.read_csv("car_actual_vs_predicted_knn.csv")

FEATURES = [
    "Price", "Maintenance Cost", "Doors", "Persons", "Luggage Boot", "Security"
]

# Step 3: Define Tabs for better organization
tab1, tab2, tab3 =st.tabs(["Real Dataset", "Model Result","New Prediction" ])

# Section: Dataset Exploration
with tab1:
    st.header("Original Car Dataset")
    st.write(f"Total records: {len(dataset)}")
    st.write(f"Features: {len(FEATURES)}")
    st.dataframe(dataset)

# Section: Model Evaluation (Testing phase results)
with tab2:
    st.header("Model performance on Test Set")

    # Calculate metrics using 'Actual' and 'Predicted' columns from the CSV
    accuracy = accuracy_score(
        results_df['Actual'],
        results_df['Predicted']
    )

    correct = (results_df["Actual"] == results_df["Predicted"]).sum()
    incorrect = len(results_df) - correct

    cm = confusion_matrix(
        results_df["Actual"],
        results_df["Predicted"]
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{accuracy:.1%}")
    col2.metric("Total Samples", len(results_df))
    col3.metric("Correct Predictions", correct)
    col4.metric("Incorrect Predictions", incorrect)

    st.subheader("Confusion Matrix & Classification Report")

    col_cm, col_cr = st.columns(2)

    with col_cm:
        st.write("Confusion Matrix")
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

    with col_cr:
        st.write("Classification Report")
        report = classification_report(
            results_df['Actual'],
            results_df["Predicted"],
            output_dict=True
        )
        report_df = pd.DataFrame(report).transpose().round(2)
        st.dataframe(report_df)


# Section: New Instance Inference
with tab3:
    st.header("Predict car for new car")
    st.write("Enter car data manually or use the example value")

    example_car = {
        "Price": 52000,
        "Maintenance Cost": "vhigh",
        "Doors": "2",
        "Persons": "2",
        "Luggage Boot": "small",
        "Security": "low"
    }

    user_input = {}

    for feature, example_val in example_car.items():

        if feature == "Price":
            user_input[feature] = st.number_input(
                feature,
                min_value=0,
                value=int(example_val),
                step=1000
            )

        elif feature == "Maintenance Cost":
            user_input[feature] = st.selectbox(
                feature,
                ["vhigh", "high", "med", "low"],
                index=["vhigh", "high", "med", "low"].index(example_val)
            )

        elif feature == "Doors":
            user_input[feature] = st.selectbox(
                feature,
                ["2", "3", "4", "5more"],
                index=["2", "3", "4", "5more"].index(example_val)
            )

        elif feature == "Persons":
            user_input[feature] = st.selectbox(
                feature,
                ["2", "4", "more"],
                index=["2", "4", "more"].index(example_val)
            )

        elif feature == "Luggage Boot":
            user_input[feature] = st.selectbox(
                feature,
                ["small", "med", "big"],
                index=["small", "med", "big"].index(example_val)
            )

        elif feature == "Security":
            user_input[feature] = st.selectbox(
                feature,
                ["low", "med", "high"],
                index=["low", "med", "high"].index(example_val)
            )

    # Execution of the prediction when the button is pressed
    if st.button("Predict"):
        
        # 1. Convert user inputs into a DataFrame
        patient_df = pd.DataFrame([user_input])
        
        
        # 2. Scale/Transform the new data using the pre-fitted transformer
        patient_scaled = preprocessor.transform(patient_df)

        # 3. Use the model to predict the category
        prediction = knn_model.predict(patient_scaled)[0]

        st.subheader("Prediction Result")
        patient_df["Prediction"] = [prediction]
        st.session_state.dataframe = pd.concat([st.session_state.dataframe, patient_df], ignore_index=True)
        if prediction == "unacc":
            st.error("Car is NOT acceptable ")
        else:
            st.success(f"Car Evaluation: {prediction.upper()} ")

    with st.expander("Show entered values"):
            st.dataframe(st.session_state.dataframe)

