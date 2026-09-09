import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


@st.cache_resource
def load_pipeline():
    return joblib.load(os.path.join(BASE_DIR, "..", "src", "property_price_pipeline.pkl"))


@st.cache_data
def load_area_lookup():
    return pd.read_csv(os.path.join(BASE_DIR, "..", "src", "area_median_log_price.csv"), index_col=0)


st.set_page_config(page_title="BD Property Price Predictor", page_icon="🏠", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: radial-gradient(circle at top left, #14161f 0%, #0b0c12 60%);
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #14161f 0%, #0b0c12 100%);
    border-right: 1px solid rgba(201, 162, 39, 0.25);
}

h1 {
    font-family: 'Playfair Display', serif;
    background: linear-gradient(90deg, #E8C15A 0%, #C9A227 60%, #9c7a1a 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 700 !important;
    letter-spacing: 0.5px;
}

h2, h3, h4 {
    font-family: 'Playfair Display', serif;
    color: #E8C15A !important;
}

p, span, label, .stMarkdown, .stCaption {
    color: #d8d2c2 !important;
}

section[data-testid="stSidebar"] h2 {
    color: #E8C15A !important;
    border-bottom: 1px solid rgba(201, 162, 39, 0.35);
    padding-bottom: 0.4em;
    margin-bottom: 0.8em;
}

div[data-baseweb="select"] > div, .stNumberInput input {
    background-color: #1b1e29 !important;
    border: 1px solid rgba(201, 162, 39, 0.35) !important;
    border-radius: 10px !important;
    color: #f1ead8 !important;
}

.stButton>button {
    background: linear-gradient(90deg, #C9A227 0%, #9c7a1a 100%);
    color: #14161f;
    border-radius: 10px;
    padding: 0.7em 1.5em;
    font-weight: 700;
    border: none;
    letter-spacing: 0.3px;
    box-shadow: 0 4px 14px rgba(201, 162, 39, 0.25);
    transition: all 0.2s ease-in-out;
}
.stButton>button:hover {
    background: linear-gradient(90deg, #E8C15A 0%, #C9A227 100%);
    box-shadow: 0 6px 18px rgba(201, 162, 39, 0.4);
    transform: translateY(-1px);
}

[data-testid="stMetric"] {
    background: linear-gradient(145deg, #1b1e29, #14161f);
    border: 1px solid rgba(201, 162, 39, 0.3);
    border-radius: 14px;
    padding: 1.2em 1em;
    box-shadow: 0 4px 16px rgba(0,0,0,0.35);
}
[data-testid="stMetricValue"] {
    color: #E8C15A !important;
    font-size: 1.9rem !important;
    font-weight: 700 !important;
}
[data-testid="stMetricLabel"] {
    color: #a89b78 !important;
    text-transform: uppercase;
    font-size: 0.75rem !important;
    letter-spacing: 1px;
}

hr, [data-testid="stDivider"] {
    border-color: rgba(201, 162, 39, 0.25) !important;
}

.stAlert {
    background-color: #1b1e29 !important;
    border: 1px solid rgba(201, 162, 39, 0.3) !important;
    border-radius: 10px !important;
    color: #d8d2c2 !important;
}
</style>
""", unsafe_allow_html=True)

pipeline = load_pipeline()
area_median_lookup = load_area_lookup()

CITY_AREAS = {
    'dhaka': ['Adabor', 'Aftab Nagar', 'Agargaon', 'Badda', 'Banani', 'Banani DOHS',
        'Banasree', 'Banglamotors', 'Bangshal', 'Baridhara', 'Baridhara DOHS', 'Bashabo',
        'Bashundhara R-A', 'Cantonment', 'Dakshin Khan', 'Demra', 'Dhanmondi', 'Dumni',
        'Eskaton', 'Gulshan', 'Hatirpool', 'Hazaribag', 'Ibrahimpur', 'Joar Sahara',
        'Kachukhet', 'Kafrul', 'Kalabagan', 'Kathalbagan', 'Keraniganj', 'Khilgaon',
        'Khilkhet', 'Kuril', 'Lalbagh', 'Lalmatia', 'Maghbazar', 'Malibagh', 'Mirpur',
        'Mohakhali', 'Mohammadpur', 'Motijheel', 'Mugdapara', 'Nadda', 'Niketan',
        'Nikunja', 'North Shahjahanpur', 'Purbachal', 'Rampura', 'Savar', 'Shahjahanpur',
        'Shantinagar', 'Shegunbagicha', 'Shiddheswari', 'Shyamoli', 'Shyampur',
        'Sutrapur', 'Tejgaon', 'Turag', 'Uttar Khan', 'Uttara'],
    'chattogram': ['10 No. North Kattali Ward', '11 No. South Kattali Ward',
        '15 No. Bagmoniram Ward', '16 No. Chawk Bazaar Ward', '22 No. Enayet Bazaar Ward',
        '29 No. West Madarbari Ward', '30 No. East Madarbari Ward', '31 No. Alkoron Ward',
        '32 No. Andarkilla Ward', '33 No. Firingee Bazaar Ward', '36 Goshail Danga Ward',
        '4 No Chandgaon Ward', '7 No. West Sholoshohor Ward', '9 No. North Pahartali Ward',
        'Bakalia', 'Bayazid', 'Double Mooring', 'East Nasirabad', 'Halishahar', 'Hathazari',
        'Jalalabad Housing Society', 'Jamal Khan', 'Kazir Dewri', 'Khulshi', 'Kotwali',
        'Lal Khan Bazaar', 'Muradpur', 'Panchlaish', 'Patenga', 'Railway Colony',
        'Sholokbahar', 'South Khulsi'],
    'cumilla': ['Ashoktala', 'Bagichagaon', 'Chotora', 'Jhautola', 'Kandirpar',
        'Moghultoli', 'Monohorpur', 'Race Course', 'Thakur Para'],
    'gazipur': ['Chandra', 'Gazipur Sadar Upazila', 'Kaliakair', 'Kapasia', 'Sreepur'],
    'narayanganj-city': ['Demra', 'Fatulla', 'Narayanganj', 'Shiddhirganj']
}

st.title("🏠 Bangladesh Property Price Predictor")
st.write("Estimate a property's sale price and see exactly which factors drive that price.")
st.divider()

with st.sidebar:
    st.header("Property Details")

    city = st.selectbox("City", list(CITY_AREAS.keys()))
    area = st.selectbox("Area", CITY_AREAS[city])

    bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10, value=3)
    bathrooms = st.number_input("Bathrooms", min_value=1, max_value=10, value=2)
    floor_area = st.number_input("Floor Area (sqft)", min_value=200, max_value=10000, value=1200)
    occupancy = st.selectbox("Occupancy Status", ["vacant", "occupied"])
    is_vacant = occupancy == "vacant"

    predict_clicked = st.button("Predict Price", use_container_width=True)

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