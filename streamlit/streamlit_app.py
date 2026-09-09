import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent))

from constants import (
    MAKE_OPTS,
    MAKE_TYPE_MAP,
    GENERIC_TYPE_FALLBACK,
    ORIGIN_OPTS,
    COLOR_OPTS,
    OPTIONS_OPTS,
    FUEL_TYPE_OPTS,
    GEAR_TYPE_OPTS,
    REGION_OPTS,
)
from model_helper import load_model, predict_price
from validation import validate_input, RAW_REQUIRED_COLS

MAE_SAR = 13849

st.set_page_config(page_title="Used Car Price Predictor", page_icon="🚗", layout="centered")

st.title("🚗 Used Car Price Prediction (Saudi Arabia)")
st.caption(
    "Estimasi harga mobil bekas berdasarkan spesifikasi. "
    "Model: XGBoost Regressor — MAE ≈ 13.728 SAR, RMSE ≈ 30.517 SAR, R² ≈ 0.822 (data test)."
)

with st.expander("ℹ️ Tentang aplikasi ini & cara kerjanya", expanded=True):
    st.markdown(
        """
**Apa ini?**
Aplikasi ini memprediksi estimasi harga mobil bekas di pasar Arab Saudi berdasarkan
spesifikasi mobil (merek, tipe, tahun, jarak tempuh, dll). Model dilatih dari data
listing ± 8.000 mobil bekas yang di-scrape dari **Syarah.com** (2021).

**Cara kerjanya**
1. Isi spesifikasi mobil (atau upload CSV untuk banyak mobil sekaligus).
2. Data mentah dilewatkan ke *preprocessing pipeline* yang sama persis dengan yang
   dipakai saat training — mengubah `Year` jadi usia mobil, meng-encode kolom
   kategorikal (Make/Type/Region berdasarkan frekuensi kemunculan), menstabilkan
   outlier & skewness pada `Mileage`, dsb.
3. Data yang sudah diproses masuk ke model **XGBoost Regressor** yang sudah dilatih
   sebelumnya (`final_model_catboost.joblib`).
4. Model memprediksi harga dalam skala log (`log1p`), lalu hasilnya dikembalikan ke
   skala SAR asli (`expm1`) sebelum ditampilkan sebagai rentang (± MAE model).

**Keterbatasan**
Model ini cenderung kurang akurat untuk mobil-mobil mewah / harga sangat tinggi,
karena datanya didominasi mobil dengan harga menengah ke bawah. Anggap hasil
prediksi sebagai *estimasi kasar*, bukan harga pasti.
        """
    )


@st.cache_resource
def get_model():
    return load_model()


model = get_model()

tab1, tab2 = st.tabs(["🔍 Single Prediction", "📄 Batch Upload (CSV)"])

# ---------------------------------------------------------------------------
# TAB 1: Single prediction via form
# ---------------------------------------------------------------------------
with tab1:
    st.subheader("Input Spesifikasi Mobil")

    # Make dipilih di luar kolom form utama supaya opsi Type bisa langsung
    # ter-filter sesuai Make (Type per Make jumlahnya bisa puluhan).
    make = st.selectbox("Make", MAKE_OPTS)
    type_options = MAKE_TYPE_MAP.get(make, []) + [GENERIC_TYPE_FALLBACK]

    col1, col2 = st.columns(2)

    with col1:
        car_type = st.selectbox("Type", type_options)
        if car_type == GENERIC_TYPE_FALLBACK:
            car_type = st.text_input("Ketik Type manual", value=make)

        year = st.number_input(
            "Year",
            min_value=1980,
            max_value=2022,
            value=2018,
            step=1,
            help=(
                "Model dilatih dengan data scrape tahun 2021 (reference_year=2021). "
                "Tahun 2021-2022 dianggap 'mobil baru' (usia 0), jadi tidak ada "
                "bedanya prediksi antara tahun 2021 dan tahun setelahnya."
            ),
        )
        origin = st.selectbox("Origin", ORIGIN_OPTS)
        color = st.selectbox("Color", COLOR_OPTS)
        options_level = st.selectbox("Options", OPTIONS_OPTS, index=1)

    with col2:
        engine_size = st.number_input(
            "Engine Size (L)", min_value=1.0, max_value=9.0, value=2.0, step=0.1
        )
        fuel_type = st.selectbox("Fuel Type", FUEL_TYPE_OPTS)
        gear_type = st.selectbox("Gear Type", GEAR_TYPE_OPTS)
        mileage = st.number_input("Mileage (km)", min_value=0, value=100000, step=1000)
        region = st.selectbox("Region", REGION_OPTS)

    if st.button("Predict", type="primary"):
        raw_df = pd.DataFrame([{
            "Make": make,
            "Type": car_type,
            "Year": year,
            "Origin": origin,
            "Color": color,
            "Options": options_level,
            "Engine_Size": engine_size,
            "Fuel_Type": fuel_type,
            "Gear_Type": gear_type,
            "Mileage": mileage,
            "Region": region,
        }])

        valid, msg = validate_input(raw_df)
        if not valid:
            st.error(msg)
        else:
            price = predict_price(model, raw_df)
            low, high = max(price[0] - MAE_SAR, 0), price[0] + MAE_SAR

            st.divider()
            st.metric("Estimasi Harga", f"{low:,.0f} – {high:,.0f} SAR")
            st.caption(f"Titik tengah estimasi: {price[0]:,.0f} SAR (± MAE model)")

            with st.expander("Lihat data input"):
                st.dataframe(raw_df)

# ---------------------------------------------------------------------------
# TAB 2: Batch prediction via CSV upload
# ---------------------------------------------------------------------------
with tab2:
    st.subheader("Upload CSV untuk Prediksi Batch")
    st.caption(f"Kolom wajib: {', '.join(RAW_REQUIRED_COLS)}")

    uploaded = st.file_uploader("Upload CSV", type="csv")

    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Gagal membaca CSV: {e}")
            df = None

        if df is not None:
            st.write(f"Preview data ({len(df)} baris):")
            st.dataframe(df)

            valid, msg = validate_input(df)
            if not valid:
                st.error(msg)
            else:
                if st.button("Predict !", type="primary", key="predict_batch"):
                    price = predict_price(model, df)

                    result_df = df.copy()
                    result_df.insert(0, "row_id", range(1, len(result_df) + 1))
                    result_df["predicted_price_low_sar"] = (price - MAE_SAR).clip(lower=0).round(0)
                    result_df["predicted_price_high_sar"] = (price + MAE_SAR).round(0)

                    st.success(f"Berhasil memprediksi {len(result_df)} baris data.")
                    st.dataframe(result_df)

                    st.download_button(
                        "⬇️ Download hasil prediksi",
                        result_df.to_csv(index=False).encode("utf-8"),
                        "prediction_result.csv",
                        "text/csv",
                    )

st.divider()
st.caption(
    "Catatan: model cenderung kurang presisi untuk mobil mewah/harga sangat tinggi "
    "(lihat bagian Limitations di notebook) — anggap hasil ini sebagai estimasi, bukan harga pasti."
)
