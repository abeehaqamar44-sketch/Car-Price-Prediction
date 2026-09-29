import streamlit as st
import pandas as pd
import joblib
import os

# -----------------------------
# PAGE CONFIGURATION
# -----------------------------
st.set_page_config(
    page_title="Car Price Prediction",
    page_icon="🚗",
    layout="wide"
)

# -----------------------------
# LOAD MODEL AND FEATURES
# -----------------------------
@st.cache_resource
def load_model():
    model = joblib.load("car_price_model.pkl")
    feature_columns = joblib.load("feature_columns.pkl")

    return model, feature_columns


model, feature_columns = load_model()

# -----------------------------
# LOAD DATASET
# -----------------------------
@st.cache_data
def load_data():
    data = pd.read_csv(
        "data/OLX_Car_Data_CSV.csv",
        encoding="latin1"
    )
    return data


data = load_data()

# -----------------------------
# TITLE
# -----------------------------
st.title("🚗 Car Price Prediction System")

st.write(
    "Machine Learning based application for predicting the price of a used car."
)

st.divider()

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select a page:",
    [
        "🏠 Home",
        "💰 Price Prediction",
        "📊 Data Analysis",
        "ℹ️ About Project"
    ]
)

# =========================================================
# HOME
# =========================================================

if page == "🏠 Home":

    st.header("Welcome to Car Price Prediction")

    st.write(
        """
        This application uses a **Random Forest Regression** machine learning
        model to estimate the price of a used car.

        Enter the details of a car on the **Price Prediction** page
        to get an estimated price.
        """
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Dataset Records",
            f"{len(data):,}"
        )

    with col2:
        st.metric(
            "Features",
            "8"
        )

    with col3:
        st.metric(
            "Model",
            "Random Forest"
        )

    st.info(
        "Use the sidebar to navigate between prediction, analysis and project information."
    )


# =========================================================
# PRICE PREDICTION
# =========================================================

elif page == "💰 Price Prediction":

    st.header("💰 Predict Car Price")

    st.write("Enter the details of your car below.")

    col1, col2 = st.columns(2)

    with col1:

        brands = sorted(
            data["Brand"].dropna().astype(str).unique().tolist()
        )

        brand = st.selectbox(
            "Brand",
            brands
        )

        conditions = sorted(
            data["Condition"].dropna().astype(str).unique().tolist()
        )

        condition = st.selectbox(
            "Condition",
            conditions
        )

        fuels = sorted(
            data["Fuel"].dropna().astype(str).unique().tolist()
        )

        fuel = st.selectbox(
            "Fuel",
            fuels
        )

        models = sorted(
            data["Model"].dropna().astype(str).unique().tolist()
        )

        car_model = st.selectbox(
            "Model",
            models
        )

    with col2:

        cities = sorted(
            data["Registered City"].dropna().astype(str).unique().tolist()
        )

        registered_city = st.selectbox(
            "Registered City",
            cities
        )

        transactions = sorted(
            data["Transaction Type"].dropna().astype(str).unique().tolist()
        )

        transaction_type = st.selectbox(
            "Transaction Type",
            transactions
        )

        kms = st.number_input(
            "KMs Driven",
            min_value=0,
            value=50000,
            step=1000
        )

        year = st.number_input(
            "Year",
            min_value=1980,
            max_value=2026,
            value=2015,
            step=1
        )

    st.divider()

    if st.button(
        "🚗 Predict Car Price",
        use_container_width=True
    ):

        # Create user input
        input_data = pd.DataFrame({
            "Brand": [brand],
            "Condition": [condition],
            "Fuel": [fuel],
            "KMs Driven": [kms],
            "Model": [car_model],
            "Registered City": [registered_city],
            "Transaction Type": [transaction_type],
            "Year": [year]
        })

        # Categorical columns
        categorical_columns = [
            "Brand",
            "Condition",
            "Fuel",
            "Model",
            "Registered City",
            "Transaction Type"
        ]

        # One-hot encoding
        input_data = pd.get_dummies(
            input_data,
            columns=categorical_columns,
            drop_first=True
        )

        # Match training features
        input_data = input_data.reindex(
            columns=feature_columns,
            fill_value=0
        )

        # Prediction
        prediction = model.predict(input_data)[0]

        st.success(
            "Prediction completed successfully!"
        )

        st.metric(
            "Estimated Car Price",
            f"PKR {prediction:,.0f}"
        )


# =========================================================
# DATA ANALYSIS
# =========================================================

elif page == "📊 Data Analysis":

    st.header("📊 Data Analysis & Visualizations")

    st.write(
        "The following graphs were created during the analysis of the dataset."
    )

    graph1 = "graphs/price_distribution.png"
    graph2 = "graphs/price_vs_year.png"
    graph3 = "graphs/price_vs_kms.png"
    graph4 = "graphs/actual_vs_predicted.png"

    if os.path.exists(graph1):
        st.subheader("1. Car Price Distribution")
        st.image(
            graph1,
            use_container_width=True
        )

    if os.path.exists(graph2):
        st.subheader("2. Car Price vs Year")
        st.image(
            graph2,
            use_container_width=True
        )

    if os.path.exists(graph3):
        st.subheader("3. Car Price vs KMs Driven")
        st.image(
            graph3,
            use_container_width=True
        )

    if os.path.exists(graph4):
        st.subheader("4. Actual vs Predicted Prices")
        st.image(
            graph4,
            use_container_width=True
        )


# =========================================================
# ABOUT PROJECT
# =========================================================

elif page == "ℹ️ About Project":

    st.header("ℹ️ About the Project")

    st.write(
        """
        ### Car Price Prediction using Machine Learning

        This project predicts the estimated price of used cars using
        a **Random Forest Regression** model.

        ### Technologies Used

        - Python
        - Pandas
        - Scikit-learn
        - Matplotlib
        - Streamlit
        - Random Forest Regression

        ### Dataset

        The dataset contains information about used cars including:

        - Brand
        - Condition
        - Fuel
        - KMs Driven
        - Model
        - Registered City
        - Transaction Type
        - Year
        - Price

        ### Model Performance

        The improved Random Forest model achieved:

        **R² Score: 0.8181**

        **MAE: 132,125.66 PKR**

        **RMSE: 273,794.48 PKR**
        """
    )

    st.success(
        "Car Price Prediction System is ready!"
    )
