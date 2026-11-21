from pathlib import Path
import requests
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from utils.prediction import (
    load_model_artifacts,
    predict_food_waste,
    validate_input_data,
)
from utils.recommendation import recommend_optimal_quantity


# ---------------------------------------------------------------------
# Page configuration and theme
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="SmartFoodAI | Food Waste Intelligence",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp { background: #0b1120; color: #f4f7fb; }
    [data-testid="stSidebar"] { background: #101a2d; border-right: 1px solid #26364d; }
    .block-container { padding-top: 1.6rem; padding-bottom: 2.5rem; max-width: 1500px; }
    h1, h2, h3 { color: #f4f7fb !important; }
    p, label, .stCaption { color: #b9c8dc; }
    div[data-testid="stMetric"] {
        background: #1d2a3e; border: 1px solid #30435d;
        padding: 16px; border-radius: 13px;
    }
    div[data-testid="stMetricLabel"] { color: #9db7d5; }
    div[data-testid="stMetricValue"] { color: #36baff; }
    div.stButton > button, div.stFormSubmitButton > button {
        background: #0fae91; color: white; border: 0;
        border-radius: 9px; font-weight: 700; min-height: 2.7rem;
    }
    div.stButton > button:hover, div.stFormSubmitButton > button:hover {
        background: #0b8d76; color: white; border: 0;
    }
    [data-testid="stExpander"] {
        background: #142137; border: 1px solid #30435d; border-radius: 10px;
    }
    hr { border-color: #30435d; }
    .hero {
        padding: 1.4rem 1.5rem; border: 1px solid #30435d;
        border-radius: 16px; background: linear-gradient(120deg,#172941,#101a2d);
        margin-bottom: 1rem;
    }
    .muted { color: #9fb3cc; }
    .small-note { color: #9fb3cc; font-size: .85rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------
# Load the exact project artifacts and dataset
# ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"


@st.cache_resource
def load_assets():
    pipeline, metadata = load_model_artifacts()
    model_path = MODEL_DIR / "random_forest_model.joblib"
    preprocessor_path = MODEL_DIR / "preprocessor.joblib"
    rf_model = joblib.load(model_path) if model_path.exists() else pipeline.named_steps["regressor"]
    preprocessor = (
        joblib.load(preprocessor_path)
        if preprocessor_path.exists()
        else pipeline.named_steps["preprocessor"]
    )
    return pipeline, metadata, rf_model, preprocessor


@st.cache_data
def load_dataset():
    cleaned = DATA_DIR / "cleaned_food_wastage_data.csv"
    raw = DATA_DIR / "food_wastage_data.csv"
    path = cleaned if cleaned.exists() else raw
    if not path.exists():
        raise FileNotFoundError(
            "Dataset not found. Expected data/cleaned_food_wastage_data.csv "
            "or data/food_wastage_data.csv."
        )
    df = pd.read_csv(path)
    if "Wastage Rate" not in df.columns and {
        "Wastage Food Amount", "Quantity of Food"
    }.issubset(df.columns):
        df["Wastage Rate"] = (
            df["Wastage Food Amount"] / df["Quantity of Food"].clip(lower=0.001) * 100
        )
    if "Food per Guest" not in df.columns and {
        "Quantity of Food", "Number of Guests"
    }.issubset(df.columns):
        df["Food per Guest"] = (
            df["Quantity of Food"] / df["Number of Guests"].clip(lower=1)
        )
    if "Waste per Guest" not in df.columns and {
        "Wastage Food Amount", "Number of Guests"
    }.issubset(df.columns):
        df["Waste per Guest"] = (
            df["Wastage Food Amount"] / df["Number of Guests"].clip(lower=1)
        )
    return df


try:
    PIPELINE, METADATA, RF_MODEL, PREPROCESSOR = load_assets()
    DATA = load_dataset()
    ASSET_ERROR = None
except Exception as exc:
    PIPELINE = METADATA = RF_MODEL = PREPROCESSOR = DATA = None
    ASSET_ERROR = str(exc)


# ---------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------
def metric_value(key, fallback="N/A"):
    metrics = METADATA.get("metrics", {}) if METADATA else {}
    value = metrics.get(key, fallback)
    return value


def make_input_form(form_key="food_input"):
    """Build a form using the exact category values saved during training."""
    options = METADATA.get("categorical_options", {})
    ranges = METADATA.get("numerical_ranges", {})

    def choices(column):
        vals = options.get(column, [])
        return list(vals) if len(vals) else ["Not available"]

    guest_range = ranges.get(
        "Number of Guests", {"min": 1, "max": 1000, "default": 100}
    )
    quantity_range = ranges.get(
        "Quantity of Food", {"min": 1, "max": 2000, "default": 300}
    )

    st.markdown("#### Event and food details")
    with st.form(form_key):
        left, right = st.columns(2)
        with left:
            food_type = st.selectbox("Type of Food", choices("Type of Food"))
            guests = st.number_input(
                "Number of Guests",
                min_value=max(1, int(guest_range["min"])),
                max_value=max(1, int(guest_range["max"])),
                value=max(1, int(guest_range["default"])),
                step=1,
            )
            event_type = st.selectbox("Event Type", choices("Event Type"))
            quantity = st.number_input(
                "Quantity of Food (dataset units)",
                min_value=max(1.0, float(quantity_range["min"])),
                max_value=max(1.0, float(quantity_range["max"])),
                value=max(1.0, float(quantity_range["default"])),
                step=1.0,
            )
            storage = st.selectbox("Storage Conditions", choices("Storage Conditions"))
        with right:
            purchase = st.selectbox("Purchase History", choices("Purchase History"))
            season = st.selectbox("Seasonality", choices("Seasonality"))
            preparation = st.selectbox("Preparation Method", choices("Preparation Method"))
            location = st.selectbox(
                "Geographical Location", choices("Geographical Location")
            )
            pricing = st.selectbox("Pricing", choices("Pricing"))

        submitted = st.form_submit_button(
            "Run Food Waste Analysis", use_container_width=True
        )

    values = {
        "Type of Food": food_type,
        "Number of Guests": int(guests),
        "Event Type": event_type,
        "Quantity of Food": float(quantity),
        "Storage Conditions": storage,
        "Purchase History": purchase,
        "Seasonality": season,
        "Preparation Method": preparation,
        "Geographical Location": location,
        "Pricing": pricing,
    }
    return values, submitted


def show_prediction_result(result):
    st.markdown("### Prediction result")
    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted food waste", f"{result['predicted_waste']:.2f} units")
    c2.metric("Wastage percentage", f"{result['wastage_percentage']:.2f}%")
    c3.metric("Waste per guest", f"{result['waste_per_guest']:.3f} units")

    color = result["risk_color"]
    st.markdown(
        f"""
        <div style="background:#17263a;padding:16px 20px;border-radius:12px;
                    border-left:6px solid {color};margin:12px 0;">
          <div style="font-size:1.2rem;font-weight:700;color:{color};">
            {result['risk_level']}
          </div>
          <div style="color:#c5d3e5;margin-top:5px;">
            {result['risk_description']}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    quantity = result["quantity_of_food"]
    remaining = max(0.0, quantity - result["predicted_waste"])
    chart = pd.DataFrame(
        {"Quantity": [result["predicted_waste"], remaining]},
        index=["Predicted waste", "Not predicted as waste"],
    )
    st.bar_chart(chart)
    st.caption(
        "Risk levels are project-defined heuristics: below 6% = Low, "
        "6–10% = Medium, above 10% = High. They are not official standards."
    )


def get_weather(city):
    """Fetch current weather from Open-Meteo; no API key is required."""
    geo_url = "https://geocoding-api.open-meteo.com/v1/search"
    geo_response = requests.get(
        geo_url,
        params={"name": city, "count": 1, "language": "en", "format": "json"},
        timeout=15,
    )
    geo_response.raise_for_status()
    locations = geo_response.json().get("results", [])
    if not locations:
        raise ValueError("Location not found. Check the city name and try again.")

    place = locations[0]
    weather_url = "https://api.open-meteo.com/v1/forecast"
    weather_response = requests.get(
        weather_url,
        params={
            "latitude": place["latitude"],
            "longitude": place["longitude"],
            "current": (
                "temperature_2m,relative_humidity_2m,precipitation,"
                "wind_speed_10m"
            ),
            "timezone": "auto",
        },
        timeout=15,
    )
    weather_response.raise_for_status()
    return place, weather_response.json().get("current", {})


def page_header(title, subtitle):
    st.markdown(
        f"""
        <div class="hero">
          <h1 style="margin:0 0 .4rem 0;">{title}</h1>
          <div class="muted">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------
st.sidebar.markdown(
    """
    <div style="padding:12px 4px 18px 4px;">
      <h2 style="color:#35baff;margin-bottom:4px;">🍽️ SmartFoodAI</h2>
      <div style="color:#a9bdd6;">Intelligent Waste Prediction & Optimization</div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    [
        "Home Dashboard",
        "Food Waste Prediction",
        "Smart Recommendations",
        "AI Explainability (SHAP)",
        "Weather Intelligence",
        "Data Analytics & EDA",
        "About Project",
    ],
)

st.sidebar.markdown("---")
if ASSET_ERROR:
    st.sidebar.error("Model assets not loaded")
    st.sidebar.caption(ASSET_ERROR)
else:
    st.sidebar.success("Model Engine: Ready")
    st.sidebar.write("**Algorithm:** Random Forest Regressor")
    st.sidebar.write(f"**Test R²:** {float(metric_value('test_r2', 0)):.3f}")

st.sidebar.markdown("---")
st.sidebar.caption("SmartFoodAI | College ML Project")
st.sidebar.caption("Built with Python, Scikit-learn & Streamlit")


if ASSET_ERROR:
    st.error("The application could not load its model or dataset.")
    st.code(ASSET_ERROR)
    st.info(
        "Check that the models/ and data/ folders exist, then run "
        "`python train_model.py` from the project root."
    )
    st.stop()


# ---------------------------------------------------------------------
# 1. Home dashboard
# ---------------------------------------------------------------------
if page == "Home Dashboard":
    page_header(
        "SmartFoodAI Intelligence Dashboard 🌿",
        "Data-driven food waste prediction, risk assessment and operational insights.",
    )

    sample_count = len(DATA)
    test_r2 = float(metric_value("test_r2", 0))
    test_mae = float(metric_value("test_mae", 0))
    baseline_rate = (
        float(DATA["Wastage Rate"].mean())
        if "Wastage Rate" in DATA.columns
        else 0.0
    )

    a, b, c, d = st.columns(4)
    a.metric("Dataset samples", f"{sample_count:,}")
    b.metric("Model test R²", f"{test_r2 * 100:.1f}%")
    c.metric("Mean Absolute Error", f"{test_mae:.3f} units")
    d.metric("Average dataset waste rate", f"{baseline_rate:.2f}%")

    st.markdown("## 🚀 End-to-end waste prevention workflow")
    w1, w2, w3 = st.columns(3)
    with w1:
        st.markdown("### 1. ML prediction")
        st.write("The trained Random Forest pipeline estimates expected surplus food.")
    with w2:
        st.markdown("### 2. Smart recommendation")
        st.write("Candidate preparation quantities are compared using the trained model.")
    with w3:
        st.markdown("### 3. Explainability and context")
        st.write("SHAP explains model contributions; weather is displayed as context.")

    st.markdown("---")
    st.subheader("Top model feature importances")
    importance = pd.DataFrame(METADATA.get("top_importances", []))
    if not importance.empty and {"Feature", "Importance"}.issubset(importance.columns):
        importance = importance.sort_values("Importance", ascending=True).tail(10)
        st.bar_chart(importance.set_index("Feature")["Importance"])
    else:
        st.info("Feature importance metadata is not available.")

    st.caption(
        "Metrics are loaded from the saved training metadata. Dataset average "
        "waste rate is descriptive, not a live prediction."
    )


# ---------------------------------------------------------------------
# 2. Food waste prediction
# ---------------------------------------------------------------------
elif page == "Food Waste Prediction":
    page_header(
        "Food Waste Prediction 🤖",
        "Enter event details and run inference using your trained Random Forest model.",
    )
    inputs, submitted = make_input_form("prediction_form")
    if submitted:
        try:
            result = predict_food_waste(inputs, pipeline=PIPELINE)
            st.session_state["last_inputs"] = inputs
            st.session_state["last_prediction"] = result
            show_prediction_result(result)
        except Exception as exc:
            st.error(f"Prediction could not be completed: {exc}")
    elif "last_prediction" in st.session_state:
        st.info("Showing the previous prediction. Submit the form to run a new one.")
        show_prediction_result(st.session_state["last_prediction"])


# ---------------------------------------------------------------------
# 3. Smart recommendations
# ---------------------------------------------------------------------
elif page == "Smart Recommendations":
    page_header(
        "Smart Recommendations 💡",
        "Compare preparation quantities and find the lowest predicted-waste option.",
    )
    st.warning(
        "This is a waste-minimization heuristic, not a guarantee of customer-demand "
        "or food-shortage optimization. Review the recommendation before using it."
    )
    inputs, submitted = make_input_form("recommendation_form")
    if submitted:
        try:
            with st.spinner("Evaluating candidate preparation quantities..."):
                result = recommend_optimal_quantity(inputs, pipeline=PIPELINE)
            st.session_state["last_inputs"] = inputs
            st.subheader("Recommended preparation plan")
            x, y, z = st.columns(3)
            x.metric("Current quantity", f"{result['current_quantity']:.1f}")
            y.metric("Recommended quantity", f"{result['recommended_quantity']:.1f}")
            z.metric("Potential waste reduction", f"{result['potential_waste_reduction']:.2f}")
            p, q, r = st.columns(3)
            p.metric("Current predicted waste", f"{result['current_predicted_waste']:.2f}")
            q.metric("Recommended predicted waste", f"{result['recommended_predicted_waste']:.2f}")
            r.metric("Waste reduction", f"{result['waste_reduction_percentage']:.1f}%")

            candidates = pd.DataFrame(result["candidates"])
            st.subheader("Candidate quantity comparison")
            if not candidates.empty:
                st.line_chart(
                    candidates.set_index("candidate_quantity")[
                        ["predicted_waste"]
                    ]
                )
                st.dataframe(candidates, use_container_width=True, hide_index=True)
            st.info(result["disclaimer"])
        except Exception as exc:
            st.error(f"Recommendation could not be generated: {exc}")


# ---------------------------------------------------------------------
# 4. SHAP explainability
# ---------------------------------------------------------------------
elif page == "AI Explainability (SHAP)":
    page_header(
        "AI Explainability (SHAP) 🧠",
        "Understand which transformed model features push an individual prediction up or down.",
    )
    st.caption(
        "SHAP explains how this model produced a prediction. It does not prove "
        "that a feature causes food waste."
    )
    inputs, submitted = make_input_form("shap_form")

    if submitted:
        try:
            import shap

            input_frame = validate_input_data(inputs)
            transformed = PREPROCESSOR.transform(input_frame)
            background = METADATA.get("shap_background_sample")
            if background is None:
                raise ValueError("Saved SHAP background sample is missing. Retrain the model.")

            with st.spinner("Calculating SHAP values..."):
                explainer = shap.TreeExplainer(
                    RF_MODEL,
                    data=np.asarray(background),
                    feature_perturbation="interventional",
                )
                shap_values = explainer.shap_values(np.asarray(transformed))
                if isinstance(shap_values, list):
                    values = np.asarray(shap_values[0])[0]
                else:
                    values = np.asarray(shap_values)
                    if values.ndim == 3:
                        values = values[0, :, 0]
                    elif values.ndim == 2:
                        values = values[0]

            names = METADATA.get("feature_names", [])
            if len(names) != len(values):
                names = [f"Feature {i+1}" for i in range(len(values))]
            explanation = pd.DataFrame(
                {"Feature": names, "SHAP value": values}
            )
            explanation["Absolute impact"] = explanation["SHAP value"].abs()
            explanation = explanation.sort_values("Absolute impact", ascending=False).head(15)

            prediction = predict_food_waste(inputs, pipeline=PIPELINE)
            st.metric("Predicted food waste", f"{prediction['predicted_waste']:.2f} units")
            st.subheader("Top SHAP contributions")
            st.bar_chart(explanation.set_index("Feature")["SHAP value"])
            st.dataframe(explanation, use_container_width=True, hide_index=True)
            st.caption(
                "Positive SHAP values push the model output higher; negative values "
                "push it lower, relative to the explainer's background reference."
            )
        except ImportError:
            st.error("SHAP is not installed. Run: python -m pip install shap")
        except Exception as exc:
            st.error(f"SHAP explanation could not be calculated: {exc}")
            st.info(
                "If this is a SHAP/NumPy compatibility issue, share the full error "
                "before changing package versions."
            )
    else:
        st.info("Fill the inputs and click Run Food Waste Analysis to generate an explanation.")


# ---------------------------------------------------------------------
# 5. Weather intelligence
# ---------------------------------------------------------------------
elif page == "Weather Intelligence":
    page_header(
        "Weather Intelligence 🌦️",
        "Fetch live weather context for a city using the free Open-Meteo service.",
    )
    st.info(
        "Weather is contextual information only. Your current Random Forest model "
        "was trained on event and food features, not weather variables."
    )
    with st.form("weather_form"):
        city = st.text_input("Enter city", value="Gorakhpur")
        weather_submit = st.form_submit_button("Get Current Weather", use_container_width=True)

    if weather_submit:
        if not city.strip():
            st.warning("Please enter a city name.")
        else:
            try:
                with st.spinner("Fetching current weather..."):
                    place, current = get_weather(city.strip())
                st.subheader(
                    f"Weather for {place.get('name', city)}, "
                    f"{place.get('country', '')}"
                )
                a, b, c, d = st.columns(4)
                a.metric("Temperature", f"{current.get('temperature_2m', 'N/A')} °C")
                b.metric("Humidity", f"{current.get('relative_humidity_2m', 'N/A')}%")
                c.metric("Precipitation", f"{current.get('precipitation', 'N/A')} mm")
                d.metric("Wind speed", f"{current.get('wind_speed_10m', 'N/A')} km/h")
                st.caption(f"Observation time: {current.get('time', 'Not provided')}")
                st.write(
                    "Operational note: consider local storage, transport and "
                    "food-safety procedures when weather conditions are unusual."
                )
            except requests.RequestException as exc:
                st.error(f"Weather service request failed: {exc}")
            except Exception as exc:
                st.error(f"Could not retrieve weather: {exc}")


# ---------------------------------------------------------------------
# 6. Data analytics and EDA
# ---------------------------------------------------------------------
elif page == "Data Analytics & EDA":
    page_header(
        "Data Analytics & EDA 📊",
        "Explore the food-wastage dataset used by the project.",
    )
    st.write(f"**Dataset shape:** {DATA.shape[0]:,} rows × {DATA.shape[1]} columns")
    with st.expander("View dataset preview"):
        st.dataframe(DATA.head(100), use_container_width=True)

    with st.expander("View descriptive statistics"):
        st.dataframe(DATA.describe(include="all").T, use_container_width=True)

    st.markdown("---")
    st.subheader("Food waste distribution")
    if "Wastage Food Amount" in DATA.columns:
        st.bar_chart(DATA["Wastage Food Amount"].value_counts().sort_index().head(30))
    if "Wastage Rate" in DATA.columns:
        st.subheader("Wastage rate distribution")
        st.line_chart(DATA["Wastage Rate"].sort_values().reset_index(drop=True))

    categorical = [
        "Type of Food",
        "Event Type",
        "Storage Conditions",
        "Purchase History",
        "Seasonality",
        "Preparation Method",
        "Geographical Location",
        "Pricing",
    ]
    available = [col for col in categorical if col in DATA.columns]
    if available:
        selected = st.selectbox("Analyze category", available)
        st.subheader(f"{selected} distribution")
        st.bar_chart(DATA[selected].value_counts())

    numeric_cols = DATA.select_dtypes(include=np.number).columns.tolist()
    if len(numeric_cols) >= 2:
        st.subheader("Numeric relationship explorer")
        x_col = st.selectbox("X-axis", numeric_cols, index=0, key="eda_x")
        y_default = numeric_cols.index("Wastage Food Amount") if "Wastage Food Amount" in numeric_cols else min(1, len(numeric_cols)-1)
        y_col = st.selectbox("Y-axis", numeric_cols, index=y_default, key="eda_y")
        st.scatter_chart(DATA[[x_col, y_col]].dropna(), x=x_col, y=y_col)

    st.caption(
        "EDA charts summarize the stored dataset. They are not live sensor readings "
        "or real-time restaurant observations."
    )


# ---------------------------------------------------------------------
# 7. About project
# ---------------------------------------------------------------------
elif page == "About Project":
    page_header(
        "About SmartFoodAI ℹ️",
        "An intelligent food-waste prediction and decision-support project.",
    )
    st.markdown(
        """
        **Objective:** estimate food wastage for an event and provide a
        waste-minimizing preparation-quantity suggestion.

        **Machine Learning algorithm:** Random Forest Regressor.

        **Input features:** type of food, number of guests, event type, quantity
        of food, storage conditions, purchase history, seasonality, preparation
        method, geographical location and pricing.

        **Target:** Wastage Food Amount.

        **Core technologies:** Python, Streamlit, Pandas, NumPy, Scikit-learn,
        SHAP, Joblib and Open-Meteo for separate weather context.
        """
    )
    st.subheader("Saved model evaluation")
    metric_df = pd.DataFrame(
        [
            ("MAE", metric_value("test_mae")),
            ("RMSE", metric_value("test_rmse")),
            ("R² score", metric_value("test_r2")),
        ],
        columns=["Metric", "Test set value"],
    )
    st.dataframe(metric_df, use_container_width=True, hide_index=True)
    st.warning(
        "The recommendation engine minimizes model-predicted waste among candidate "
        "quantities. The dataset does not record customer shortages or satisfaction, "
        "so this should not be treated as guaranteed demand planning."
    )

st.markdown("---")
st.caption(
    "SmartFoodAI • Random Forest food-waste prediction • "
    "Project-defined risk thresholds • Verify recommendations with operational judgment"
)
