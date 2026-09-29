import streamlit as st
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

model = load_model("model/breast_cancer_pred_model.keras")
scaler = joblib.load("model/scaler.pkl")

st.title("Breast Cancer Prediction App")

st.write(
    "Use either manual input or upload a CSV file "
    "to perform breast cancer prediction."
)

features = [
    "mean radius",
    "mean texture",
    "mean perimeter",
    "mean area",
    "mean smoothness",
    "mean compactness",
    "mean concavity",
    "mean concave points",
    "mean symmetry",
    "mean fractal dimension",
    "radius error",
    "texture error",
    "perimeter error",
    "area error",
    "smoothness error",
    "compactness error",
    "concavity error",
    "concave points error",
    "symmetry error",
    "fractal dimension error",
    "worst radius",
    "worst texture",
    "worst perimeter",
    "worst area",
    "worst smoothness",
    "worst compactness",
    "worst concavity",
    "worst concave points",
    "worst symmetry",
    "worst fractal dimension"
]

input_method = st.radio(
    "Select Input Method:",
    ["Manual Input", "CSV File"],
    horizontal=True
)

if input_method == "Manual Input":
    st.subheader("Enter Feature Values")
    input_data = []

    # Use two columns to make the interface more compact
    col1, col2 = st.columns(2)

    for i, feature in enumerate(features):
        if i % 2 == 0:
            with col1:
                value = st.number_input(
                    feature,
                    min_value=0.0,
                    value=0.0,
                    key=feature
                )
        else:
            with col2:
                value = st.number_input(
                    feature,
                    min_value=0.0,
                    value=0.0,
                    key=feature
                )

        input_data.append(value)

else:
    st.subheader("Upload CSV File")
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"]
    )

    if uploaded_file is not None:
        csv_data = pd.read_csv(uploaded_file)

        rename_map = {}
        for feature in features:
            if feature.startswith("mean "):
                csv_name = feature[5:] + "_mean"
            elif feature.endswith(" error"):
                csv_name = feature[:-6] + "_se"
            elif feature.startswith("worst "):
                csv_name = feature[6:] + "_worst"
            else:
                continue
            rename_map[csv_name] = feature

        csv_data = csv_data.rename(columns=rename_map)
        st.write("Preview of uploaded data:")
        st.dataframe(csv_data.head())

        # Check whether all required features exist
        missing_features = [
            feature for feature in features
            if feature not in csv_data.columns
        ]

        if missing_features:
            st.error(
                "The CSV file is missing the following features:"
            )
            st.write(missing_features)
        else:
            st.success(
                "CSV file loaded successfully. "
                f"{len(csv_data)} records found."
            )

st.divider()

if st.button("Run Predict", type="primary"):

    
    # Manual Input Prediction
    if input_method == "Manual Input":

        input_df = pd.DataFrame(
            [input_data], columns=features
        )

        input_scaled = scaler.transform(input_df)

        prediction = model.predict(
            input_scaled,
            verbose=0
        )

        probability = prediction[0][0]

        if probability >= 0.5:
            result = "Benign"
            result_probability = probability
        else:
            result = "Malignant"
            result_probability = 1 - probability

        st.subheader("Prediction Result")

        if result == "Malignant":
            st.error(f"Prediction: **{result}**")
        else:
            st.success(f"Prediction: **{result}**")

        st.write(
            f"Prediction probability for {result}: **{result_probability:.4f}**"
        )

    else:
        if uploaded_file is None:
            st.warning(
                "Please upload a CSV file first."
            )
        elif missing_features:
            st.error(
                "Prediction cannot be performed because "
                "some required features are missing."
            )
        else:
            X_new = csv_data[features]

            X_scaled = scaler.transform(X_new)

            predictions = model.predict(X_scaled, verbose=0)

            probabilities = predictions.flatten()

            results = np.where(
                probabilities >= 0.5, "Benign", "Malignant"
            )

            result_df = csv_data.copy()
            result_df["Prediction Probability"] = probabilities
            result_df["Prediction"] = results

            st.subheader("Prediction Results")
            st.dataframe(result_df)

            malignant_count = np.sum(results == "Malignant")
            benign_count = np.sum(results == "Benign")
            col1, col2 = st.columns(2)

            with col1:
                st.metric("Benign", benign_count)
            with col2:
                st.metric("Malignant", malignant_count)

            csv_output = result_df.to_csv(
                index=False
            )

            st.download_button(
                label="Download Prediction Results",
                data=csv_output,
                file_name="breast_cancer_predictions.csv",
                mime="text/csv"
            )
