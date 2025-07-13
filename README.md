# 🏠 California Housing Price Prediction

This project predicts the median house value in California districts using various housing-related features. It's powered by a Decision Tree Regressor and deployed using Streamlit.

---

## 📊 Overview

- **Problem:** Estimate housing prices based on district-level attributes.
- **Model Used:** Decision Tree Regressor
- **Deployment:** Streamlit app
- **Pipeline:** Custom preprocessing pipeline with feature engineering and encoding.

---

## 📁 Dataset

- **Source:** California Housing Dataset (from StatLib repository or Scikit-Learn)
- **Features:**
  - `longitude`, `latitude`
  - `housing_median_age`
  - `total_rooms`, `total_bedrooms`
  - `population`, `households`
  - `median_income`
  - `ocean_proximity` (categorical)
- **Target:** `median_house_value`

---

## 🛠️ Features

- Data cleaning and preprocessing pipeline
- Custom transformers (`DataFrameSelector`, `CombinedAttributesAdder`)
- DecisionTreeRegressor model
- Feature engineering (e.g., `rooms_per_household`)
- Deployment using Streamlit

---



