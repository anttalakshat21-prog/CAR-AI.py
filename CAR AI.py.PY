import os
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Page configuration
st.set_page_config(
    page_title="AutoValue AI - Used Car Price Prediction",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling (Dark Navy & White theme with Blue Accents)
st.markdown(
    """
    <style>
    .main { background-color: #f8f9fa; }
    .stHeading h1, .stHeading h2, .stHeading h3 { color: #0a192f; font-family: 'Helvetica Neue', Arial, sans-serif; }
    .card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        margin-bottom: 20px;
        border-left: 5px solid #007bff;
    }
    .metric-value { font-size: 28px; font-weight: bold; color: #007bff; }
    </style>
""",
    unsafe_allow_html=True,
)

# Initialize Session States
if "trained_models" not in st.session_state:
    st.session_state.trained_models = None
if "df" not in st.session_state:
    st.session_state.df = None
if "col_mapping" not in st.session_state:
    st.session_state.col_mapping = {}

# Navigation Sidebar
st.sidebar.title("🚗 AutoValue AI")
st.sidebar.markdown("**Project No:** 32 | **Section:** 216B")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigate",
    [
        "1. Home Dashboard",
        "2. Predict Car Price",
        "3. Upload Dataset",
        "4. Train Machine Learning Models",
        "5. Model Performance",
        "6. Data Visualization",
        "7. About the Project",
    ],
)


def normalize_column_names(df):
    mapping = {}
    cols = df.columns
    # Target Price Mapping
    for c in cols:
        if c.lower() in [
            "selling_price",
            "selling price",
            "price",
            "resale_price",
            "resale price",
        ]:
            mapping["target"] = c
            break
    # Brand/Model Mapping
    for c in cols:
        if c.lower() in [
            "brand",
            "car_name",
            "name",
            "manufacturer",
            "car name",
        ]:
            mapping["brand"] = c
            break
    # Mileage Mapping
    for c in cols:
        if c.lower() in [
            "mileage",
            "km_driven",
            "kilometers_driven",
            "km driven",
            "kilometers driven",
        ]:
            mapping["mileage"] = c
            break
    # Fuel/Engine type Mapping
    for c in cols:
        if c.lower() in [
            "fuel",
            "fuel_type",
            "engine_type",
            "fuel type",
            "engine type",
        ]:
            mapping["fuel"] = c
            break
    # Year / Age Mapping
    for c in cols:
        if c.lower() in [
            "year",
            "manufacturing_year",
            "vehicle_age",
            "age",
            "manufacturing year",
        ]:
            mapping["year"] = c
            break
    return mapping


def clean_and_parse_units(val):
    if pd.isna(val):
        return np.nan
    if isinstance(val, (int, float)):
        return float(val)
    val_str = str(val).lower().strip()
    cleaned = "".join([c for c in val_str if c.isdigit() or c == "."])
    try:
        return float(cleaned) if cleaned else np.nan
    except:
        return np.nan


# --- 1. HOME DASHBOARD ---
if page == "1. Home Dashboard":
    st.title("🚗 AutoValue AI Dashboard")
    st.subheader("Smart Used Car Price Prediction System")

    st.markdown(
        """<div class='card'>
        <h4>Welcome to the AutoValue AI Suite</h4>
        <p>This intelligent predictive engine eliminates transaction information gaps by helping users negotiate fair market valuations based on vehicle age, brand profile, and historical usage patterns.</p>
        <p><b>Academic Credentials:</b> B.Tech CSE Capstone Project #32 | Section 216B</p>
    </div>""",
        unsafe_allow_html=True,
    )
    col1, col2 = st.columns(2)
    with col1:
        st.info(
            " **Core Objective:** Estimate precise resale valuation offsets through structured data analysis pipelines. "
        )
    with col2:
        st.success(
            " **Integrated Algorithms:** Ordinary Least Squares (OLS) Linear Regression & Decoupled Decision Tree Random Forest Regressors. "
        )

# --- 3. UPLOAD DATASET ---
elif page == "3. Upload Dataset":
    st.title("📥 Upload CSV Dataset")
    uploaded_file = st.file_uploader("Choose a used car CSV file", type=["csv"])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            if df.empty:
                st.error("Uploaded file contain no valid entries.")
            else:
                st.session_state.df = df
                st.success(
                    f"Successfully loaded file containing {df.shape[0]} matrix rows!"
                )
                mapping = normalize_column_names(df)
                st.session_state.col_mapping = mapping

                st.markdown("### Feature Column Matching Settings")
                cols = df.columns.tolist()
                target_col = st.selectbox(
                    "Target Variable (Price)",
                    cols,
                    index=(
                        cols.index(mapping.get("target"))
                        if mapping.get("target") in cols
                        else 0
                    ),
                )
                brand_col = st.selectbox(
                    "Brand Name Column",
                    cols,
                    index=(
                        cols.index(mapping.get("brand"))
                        if mapping.get("brand") in cols
                        else 0
                    ),
                )
                mileage_col = st.selectbox(
                    "Mileage/KM Driven Column",
                    cols,
                    index=(
                        cols.index(mapping.get("mileage"))
                        if mapping.get("mileage") in cols
                        else 0
                    ),
                )
                fuel_col = st.selectbox(
                    "Fuel/Engine Type Column",
                    cols,
                    index=(
                        cols.index(mapping.get("fuel"))
                        if mapping.get("fuel") in cols
                        else 0
                    ),
                )
                year_col = st.selectbox(
                    "Manufacturing Year Column",
                    cols,
                    index=(
                        cols.index(mapping.get("year"))
                        if mapping.get("year") in cols
                        else 0
                    ),
                )

                st.session_state.col_mapping = {
                    "target": target_col,
                    "brand": brand_col,
                    "mileage": mileage_col,
                    "fuel": fuel_col,
                    "year": year_col,
                }

                st.subheader("Data Matrix Preview")
                st.dataframe(df.head(10))

                st.subheader("Descriptive Summary Metrics")
                st.write(df.describe())
                csv_data = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download Processed Matrix Layout",
                    data=csv_data,
                    file_name="processed_car_data.csv",
                    mime="text/csv",
                )
        except Exception as e:
            st.error(f"Error parsing dataframe stream: {e}")
    else:
        st.warning(
            "Please upload a CSV sheet file container to seed training pipelines."
        )

# --- 4. TRAIN MACHINE LEARNING MODELS ---
elif page == "4. Train Machine Learning Models":
    st.title("⚙️ Model Pipeline Control Desk")
    if st.session_state.df is None:
        st.warning(
            "Please upload a raw CSV matrix schema first under the Upload Dataset navigation panel."
        )
    else:
        df = st.session_state.df.copy()
        m = st.session_state.col_mapping

        st.markdown("### Pipeline Tuning Configuration")
        test_size = (
            st.slider("Test Set Validation Split (%)", 10, 50, 20) / 100.0
        )
        if st.button("Execute Pipeline Fitting Routine"):
            try:
                df[m["mileage"]] = df[m["mileage"]].apply(
                    clean_and_parse_units
                )
                df[m["target"]] = pd.to_numeric(
                    df[m["target"]], errors="coerce"
                )
                df[m["year"]] = pd.to_numeric(df[m["year"]], errors="coerce")
                df = df.dropna(
                    subset=[
                        m["target"],
                        m["brand"],
                        m["mileage"],
                        m["fuel"],
                        m["year"],
                    ]
                )
                df = df[df[m["target"]] > 0]
                if len(df) < 5:
                    st.error(
                        "Dataset has insufficient usable rows remaining after sanitisation filters."
                    )
                else:
                    if df[m["year"]].max() > 1900:
                        df["vehicle_age_computed"] = 2026 - df[m["year"]]
                    else:
                        df["vehicle_age_computed"] = df[m["year"]]
                    X = df[
                        [
                            m["brand"],
                            m["mileage"],
                            m["fuel"],
                            "vehicle_age_computed",
                        ]