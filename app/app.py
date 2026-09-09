import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# ── Page config (must be first Streamlit command) ────────────────────────────
st.set_page_config(page_title="BD Property Price Predictor", page_icon="🏠", layout="wide")

# ── Custom styling ─────────────────────────────────────────────────────────
st.markdown("""
<style>
.main { background-color: #f7f9fb; }
body, p, span, label, .stMarkdown { color: #C9A227 !important; }
h1, h2, h3 { color: #C9A227 !important; }
.stButton>button {
    background-color: #1e3a5f;
    color: white;
    border-radius: 8px;
    padding: 0.6em 1.5em;
    font-weight: 600;
    border: none;
}
.stButton>button:hover { background-color: #2c5282; }
[data-testid="stMetricValue"] { color: #C9A227 !important; font-size: 2rem; }
[data-testid="stMetricLabel"] { color: #C9A227 !important; }
</style>
""", unsafe_allow_html=True)

# ── Load model + lookup table (cached so it only loads once) ────────────────
@st.cache_resource
def load_pipeline():
    return joblib.load("../src/property_price_pipeline.pkl")

@st.cache_data
def load_area_lookup():
    return pd.read_csv("../src/area_median_log_price.csv", index_col=0)

pipeline = load_pipeline()
area_median_lookup = load_area_lookup()

# ── Header ───────────────────────────────────────────────────────────────────
st.title("🏠 Bangladesh Property Price Predictor")
st.write("Estimate a property's sale price and see exactly which factors drive that price.")
st.divider()

# ── Sidebar: inputs ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Property Details")

    city = st.selectbox("City", ['dhaka', 'chattogram', 'cumilla', 'narayanganj-city', 'gazipur'])

    area = st.selectbox("Area", [
        'Mirpur', 'Narayanganj', 'Gazipur Sadar Upazila', '9 No. North Pahartali Ward',
        'Bakalia', 'Double Mooring', 'Bayazid', '4 No Chandgaon Ward', 'Bashundhara R-A',
        'Khulshi', 'Halishahar', 'Uttara', 'Badda', 'Bagichagaon', 'Sholokbahar',
        'Mohammadpur', 'Dakshin Khan', '7 No. West Sholoshohor Ward', 'Muradpur',
        '15 No. Bagmoniram Ward'
    ])

    bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10, value=3)
    bathrooms = st.number_input("Bathrooms", min_value=1, max_value=10, value=2)
    floor_area = st.number_input("Floor Area (sqft)", min_value=200, max_value=10000, value=1200)
    occupancy = st.selectbox("Occupancy Status", ["vacant", "occupied"])
    is_vacant = occupancy == "vacant"

    predict_clicked = st.button("Predict Price", use_container_width=True)

# ── Main area: results ───────────────────────────────────────────────────────
if predict_clicked:
    bed_bath_ratio = bedrooms / bathrooms if bathrooms > 0 else np.nan

    if area in area_median_lookup.index:
        area_median_log_price = area_median_lookup.loc[area, 'log_price']
    else:
        area_median_log_price = area_median_lookup['log_price'].median()

    input_df = pd.DataFrame([{
        "Bedrooms": bedrooms,
        "Bathrooms": bathrooms,
        "Floor_area": floor_area,
        "is_vacant": int(is_vacant),
        "bed_bath_ratio": bed_bath_ratio,
        "area_median_log_price": area_median_log_price,
        "City": city,
        "area": area
    }])

    pred_log = pipeline.predict(input_df)[0]
    pred_price = np.expm1(pred_log)

    col1, col2, col3 = st.columns(3)
    col1.metric("Estimated Price", f"৳{pred_price:,.0f}")
    col2.metric("Price per sqft", f"৳{pred_price/floor_area:,.0f}")
    col3.metric("City", city.title())

    st.divider()
    st.subheader("Why this price? (SHAP explanation)")
    st.caption("This shows which specific factors pushed the estimate up or down for this property.")

    preprocessor = pipeline.named_steps['preprocessor']
    model = pipeline.named_steps['model']
    feature_names = preprocessor.get_feature_names_out()

    input_transformed = preprocessor.transform(input_df)
    if hasattr(input_transformed, "toarray"):
        input_transformed = input_transformed.toarray()
    input_transformed = input_transformed.astype(np.float64)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(input_transformed)
    shap_values.feature_names = list(feature_names)

    plt.close('all')
    fig = plt.figure()
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(fig)

else:
    st.info("Fill in the property details on the left and click **Predict Price** to see an estimate.")