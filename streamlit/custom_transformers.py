"""
custom_transformers.py

WAJIB ada dan didefinisikan SEBELUM joblib.load() dipanggil, karena
model CatBoost yang di-pickle menyimpan referensi ke class ini.

Berisi:
- CarAgeTransformer: mengubah kolom `Year` mentah menjadi `car_age`
  (reference_year - Year), reference_year default 2021 sesuai tahun
  data di-scrape dari Syarah.com.
- FrequencyEncoder: meng-encode kolom kategorikal berkardinalitas tinggi
  (Make, Type, Region) menjadi proporsi frekuensi kemunculan tiap
  kategori di data training. Kategori yang tidak pernah dilihat saat
  fit akan diberi nilai 0.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class CarAgeTransformer(BaseEstimator, TransformerMixin):
    """Tambah kolom car_age = reference_year - Year ke seluruh DataFrame.

    Beroperasi di awal pipeline (sebelum ColumnTransformer), jadi input
    yang diterima adalah SEMUA kolom fitur mentah, dan output yang
    dikembalikan juga harus DataFrame dengan seluruh kolom asli plus
    kolom baru `car_age`, supaya ColumnTransformer di step berikutnya
    bisa memilih kolom berdasarkan nama.
    """

    def __init__(self, reference_year: int = 2021):
        self.reference_year = reference_year

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        X["car_age"] = (self.reference_year - X["Year"].astype(int)).clip(lower=0)
        return X


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """Encode kolom kategorikal berdasarkan proporsi frekuensi di data training.

    Menyimpan `freq_maps_` sebagai list peta frekuensi per kolom, sesuai
    urutan kolom input saat fit (positional, bukan keyed by name), agar
    tetap bekerja baik input berupa DataFrame maupun array numpy (seperti
    yang diteruskan oleh ColumnTransformer/Pipeline).
    """

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.feature_names_in_ = list(X.columns) if hasattr(X, "columns") else None

        n = len(X)
        self.freq_maps_ = [
            (X.iloc[:, i].value_counts() / n).to_dict() for i in range(X.shape[1])
        ]
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        out = pd.DataFrame(index=X.index)
        for i in range(X.shape[1]):
            col_name = X.columns[i]
            freq_map = self.freq_maps_[i]
            out[col_name] = X.iloc[:, i].map(freq_map).fillna(0.0)
        return out.to_numpy()

    def get_feature_names_out(self, input_features=None):
        if self.feature_names_in_ is not None:
            return np.array(self.feature_names_in_)
        return np.array([f"freq_{i}" for i in range(len(self.freq_maps_))])
