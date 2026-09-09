"""
validation.py

Validasi input (DataFrame, satu atau banyak baris) sebelum masuk ke
model. Dipakai di streamlit_app.py untuk tab Single Prediction maupun
Batch Upload (CSV).
"""

import pandas as pd

# 11 kolom fitur mentah yang wajib ada, persis seperti X_train
# sebelum masuk preprocessing pipeline di notebook.
RAW_REQUIRED_COLS = [
    "Make",
    "Type",
    "Year",
    "Origin",
    "Color",
    "Options",
    "Engine_Size",
    "Fuel_Type",
    "Gear_Type",
    "Mileage",
    "Region",
]

MAX_YEAR = 2022  # CarAgeTransformer pakai reference_year=2021


def validate_input(df: pd.DataFrame):
    """Validasi DataFrame input.

    Return
    ------
    (valid: bool, message: str)
        valid=True dan message="" kalau tidak ada masalah.
    """
    errors = []

    missing_cols = [c for c in RAW_REQUIRED_COLS if c not in df.columns]
    if missing_cols:
        return False, f"Kolom wajib tidak ditemukan: {', '.join(missing_cols)}"

    if df[RAW_REQUIRED_COLS].isnull().any().any():
        null_cols = df[RAW_REQUIRED_COLS].columns[df[RAW_REQUIRED_COLS].isnull().any()].tolist()
        errors.append(f"Ada nilai kosong di kolom: {', '.join(null_cols)}")

    if (df["Year"] > MAX_YEAR).any():
        errors.append(f"Year tidak boleh lebih dari {MAX_YEAR}.")

    if (df["Year"] < 1980).any():
        errors.append("Year tidak boleh kurang dari 1980.")

    if (df["Mileage"] < 0).any():
        errors.append("Mileage tidak boleh negatif.")

    if (df["Engine_Size"] <= 0).any():
        errors.append("Engine_Size harus lebih besar dari 0.")

    if errors:
        return False, " | ".join(errors)

    return True, ""
