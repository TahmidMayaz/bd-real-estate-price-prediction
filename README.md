# 🏆 Bangladesh Real Estate Price Predictor

An end-to-end machine learning system that predicts Bangladesh property prices from real-estate listing data, and explains *why* — using SHAP-based interpretability to show how location, floor area, and other factors drive price.

🔗 **Live app:** [bd-real-estate-price-prediction.streamlit.app](https://bd-real-estate-price-prediction.streamlit.app/)
📊 **Dataset:** ~3,900 property listings across 5 Bangladeshi cities (Dhaka, Chattogram, Cumilla, Narayanganj, Gazipur)

---

## What it does

Given a property's city, area, size, bedrooms, bathrooms, and occupancy status, the app predicts an estimated sale price in taka — and shows a SHAP waterfall chart explaining exactly which features pushed that specific prediction up or down.

## Key findings

- **Floor area** and **area-level median price** are by far the strongest predictors of price — far more influential than bedroom or bathroom count.
- **City** matters significantly on its own — Chattogram and Dhaka listings behave differently from the smaller cities in the dataset, even after controlling for size and location.
- Best-performing model: **Random Forest Regressor**, R² = 0.84 on held-out test data, MAE ≈ ৳802,000.

## Tech stack

```
Python
├── Pandas / NumPy          → data handling
├── Matplotlib / Seaborn    → EDA visuals
├── Scikit-learn            → preprocessing pipeline, Linear/RandomForest baselines
├── XGBoost                 → gradient-boosted comparison model
├── SHAP                    → model explainability
└── Streamlit               → deployed predictor app
```

## Pipeline

```
Raw listings CSV
   ↓
Cleaning (price parsing, outlier removal, dedup)
   ↓
EDA
   ↓
Feature engineering (leakage-safe)
   ↓
Train/test split → area-median price computed from TRAIN ONLY
   ↓
Preprocessing + model pipeline (Linear / RandomForest / XGBoost)
   ↓
Evaluation (R², RMSE, MAE — in both log-space and taka)
   ↓
SHAP analysis
   ↓
Streamlit app (single saved pipeline artifact)
   ↓
Deployed on Streamlit Community Cloud
```

A key design decision: the original feature set included `price_per_sqft` and `area_median_price`, both computed directly from the target (`price`) across the full dataset — this causes **target leakage** and artificially inflates model performance. The final pipeline instead computes `area_median_log_price` from the **training split only**, then maps it onto the test set — keeping evaluation honest.

## Project structure

```
bd-real-estate/
├── data/
│   ├── raw/              # original dataset
│   └── processed/        # cleaned data, feature set, saved chart images
├── notebooks/             # step-by-step analysis notebooks
├── app/
│   └── app.py             # Streamlit app
├── src/
│   ├── property_price_pipeline.pkl    # trained preprocessing + model pipeline
│   └── area_median_log_price.csv      # area-level price lookup (train-derived)
└── requirements.txt
```

## Run it locally

```bash
git clone https://github.com/TahmidMayaz/bd-real-estate-price-prediction.git
cd bd-real-estate-price-prediction
pip install -r requirements.txt
cd app
streamlit run app.py
```

## Limitations & honest caveats

- Prices reflect listing values as collected — no inflation adjustment was applied.
- The model is unreliable for feature combinations rarely seen in training (e.g. very low bathroom count in an area where most comparable listings have 3+ bathrooms) — predictions in these edge cases should be treated with caution.
- Dataset covers 5 cities only and does not capture year-over-year market trends.

---

*Built as a portfolio project applying end-to-end ML engineering: data cleaning, leakage-safe feature engineering, model comparison, explainability, and deployment.*