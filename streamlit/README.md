# Used Car Price Prediction — Streamlit App

Aplikasi Streamlit untuk prediksi harga mobil bekas. Struktur flat: satu
file `streamlit_app.py` di root folder ini dengan `st.tabs()` (Single
Prediction / Batch Upload CSV).

Folder ini adalah bagian dari repo utama proyek (di dalam `streamlit/`),
bukan repo terpisah.

## Struktur folder

```
streamlit/
├── streamlit_app.py            # entry point, tabs Single Prediction & Batch Upload
├── constants.py                 # daftar Make/Type/Region dsb dari data training
├── custom_transformers.py       # class CarAgeTransformer dan FrequencyEncoder
├── model_helper.py              # load_model() + predict_price()
├── validation.py                # validasi input sebelum prediksi
├── requirements.txt
└── models/
    └── used_car_price_model_xgboost.joblib
```

## Langkah menjalankan

1. Pastikan `models/used_car_price_model_xgboost.joblib` sudah ada
   (kalau belum, export dari notebook lalu copy ke folder `models/` ini).
2. `pip install -r requirements.txt`
3. `streamlit run streamlit_app.py`

## Catatan

- 11 kolom fitur mentah yang wajib ada: `Make, Type, Year, Origin, Color, Options, Engine_Size, Fuel_Type, Gear_Type, Mileage, Region` — persis
  seperti `X_train` sebelum masuk pipeline di notebook.
- Tab "Batch Upload (CSV)" menerima file dengan 11 kolom di atas, menambahkan
  kolom `predicted_price_sar`, dan bisa langsung di-download hasilnya.
