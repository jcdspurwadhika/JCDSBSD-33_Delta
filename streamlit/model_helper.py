"""
model_helper.py

load_model() + predict_price(), termasuk fix injeksi
CarAgeTransformer & FrequencyEncoder ke sys.modules["__main__"]
sebelum joblib.load() (untuk hindari AttributeError saat unpickle,
karena model di-pickle dari notebook di mana kedua class ini
didefinisikan di __main__).
"""

import sys
from pathlib import Path

import joblib
import numpy as np

from custom_transformers import CarAgeTransformer, FrequencyEncoder

MODEL_PATH = Path(__file__).resolve().parent / "models" / "used_car_price_model_xgboost.joblib"


def load_model():
    # Fix: inject custom classes ke __main__ agar joblib.load() tidak error
    sys.modules["__main__"].CarAgeTransformer = CarAgeTransformer
    sys.modules["__main__"].FrequencyEncoder = FrequencyEncoder
    return joblib.load(MODEL_PATH)


def predict_price(model, X):
    """X: DataFrame dengan 11 kolom fitur mentah, return harga dalam SAR."""
    log_pred = model.predict(X)
    return np.expm1(log_pred)
