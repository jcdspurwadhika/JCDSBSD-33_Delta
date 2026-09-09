# Used Car Price Prediction - Syarah.com

Disusun oleh:

- Athiyyah Nisrina Husna
- Firsa Adam

## 1. Project Overview

Proyek ini menganalisis data mobil bekas yang di-scrape dari **Syarah.com** (marketplace mobil online di Arab Saudi, 2021) untuk membangun model *machine learning* yang dapat memprediksi harga jual mobil bekas secara akurat. Fokus utama proyek adalah **membangun model regresi harga (`Price`, dalam SAR) yang andal**, termasuk menangani distribusi harga yang skewed dan outlier pada mobil-mobil mewah.

## 2. Problem Statement

Listing mobil bekas dengan status "Negotiable" (harga tidak dicantumkan penjual) menyulitkan calon pembeli maupun penjual untuk memperkirakan harga wajar di awal, sehingga proses negosiasi jadi kurang efisien. Estimasi harga otomatis berbasis spesifikasi kendaraan dapat membantu kedua pihak mendapatkan titik awal harga yang lebih objektif sebelum negosiasi lebih lanjut.

## 3. Project Objectives

- Memprediksi estimasi harga wajar untuk listing mobil dengan status "Negotiable" (harga tidak dicantumkan penjual), menggunakan pola dari data mobil yang harganya sudah diketahui.
- Meningkatkan akurasi estimasi harga berdasarkan spesifikasi kendaraan (merek, tipe, tahun, jarak tempuh, dll).
- Menyediakan metrik evaluasi yang relevan (MAE, RMSE, R²) agar estimasi harga untuk listing negotiable dapat dipercaya dan dipertanggungjawabkan.

## 4. Data Sources

- [UsedCarsSA_Unclean_EN.csv](data/raw/UsedCarsSA_Unclean_EN.csv) – Data mentah hasil scraping Syarah.com (2021), 8.248 baris, 15 kolom.
- [UsedCarsSA_Clean_FINAL.csv](data/processed/UsedCarsSA_Clean_FINAL.csv) – Data yang sudah dibersihkan, siap untuk EDA & modeling.
- Sumber asli dataset: [Kaggle - Used Cars Dataset (Saudi Arabia)](https://www.kaggle.com/datasets/turkibintalib/saudi-arabia-used-cars-dataset)

## 5. Technologies Used

- Programming Language: Python (Pandas, NumPy)
- Machine Learning: Scikit-Learn, XGBoost, LightGBM, CatBoost, Feature-engine
- Visualization: Matplotlib, Seaborn
- Deployment: Streamlit
- Version Control: Git & GitHub
- Others: Jupyter Notebook, Joblib

## 6. Analysis Approach

Analisis dilakukan melalui alur kerja data science end-to-end, meliputi:

### 6.1  Data Preparation

- Pembersihan data (menghapus baris dengan `Price = 0`, duplikat, dan anomali lain).
- Feature engineering: `car_age` dari `Year` menggunakan `CarAgeTransformer` (referensi tahun 2021).

### 6.2 Preprocessing

- `SimpleImputer` (median) + `StandardScaler` untuk fitur numerik (`Mileage`, `Engine_Size`, `car_age`).
- `OneHotEncoder(drop="first")` untuk kategorikal nominal (`Fuel_Type`, `Origin`, `Color`, `Gear_Type`).
- `OrdinalEncoder` untuk `Options` (urutan `Standard` < `Semi Full` < `Full`).
- `FrequencyEncoder` (custom) untuk kategorikal berkardinalitas tinggi (`Make`, `Type`, `Region`).
- Target (`Price`) di-log-transform (`log1p`) setelah train-test split, sebelum masuk preprocessing/modeling, untuk mengatasi skewness.

### 6.3 Modeling

- 5-fold KFold cross-validation, metrik dihitung balik ke skala SAR (`expm1`) agar comparable.
- Benchmarking 9 model (Linear Regression, KNN, Random Forest, Decision Tree, XGBoost, LightGBM, CatBoost, AdaBoost, Lasso). CatBoost tampil terbaik di baseline (MAE 14.098 SAR), diikuti XGBoost dan LightGBM.
- Tiga kandidat teratas (CatBoost, XGBoost, LightGBM) di-tuning dengan `RandomizedSearchCV`; **XGBoost** terpilih sebagai model final setelah tuning.

## 7. Evaluation Metrics

Evaluasi model menggunakan Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), dan R² — dihitung pada skala harga asli (SAR), bukan skala log, agar comparable dengan kebutuhan bisnis. Success criteria proyek: R² ≥ 0,80.

Model final (XGBoost hasil tuning) pada data testing:

| Metrik | Nilai      |
| ------ | ---------- |
| MAE    | 13.728 SAR |
| RMSE   | 30.517 SAR |
| R²    | 0,823      |

## 8. Structure Repository

```text
├── dashboard           <- Dashboard hasil analisis/visualisasi.
│
├── data
│   ├── processed       <- Dataset final & bersih, siap untuk modeling (UsedCarsSA_Clean_EN.csv).
│   └── raw             <- Data mentah asli hasil scraping (UsedCarsSA_Unclean_EN.csv), tidak diubah.
│
├── models              <- Model terlatih & tersimpan (used_car_price_model_xgboost.joblib).
│
├── notebooks           <- Jupyter notebook analisis.
|
├── reference           <- Referensi eksternal.
|
├── streamlit           <- Aplikasi Streamlit untuk prediksi harga (lihat README di dalamnya).
│
├── README.md           <- Top-level README untuk proyek ini.
│
└── requirements.txt    <- Dependensi Python untuk reproduksi environment analisis.
```

## 9. Summary of Finding

### 9.1 Business Insight

Model final (XGBoost) memprediksi harga mobil bekas dengan MAE 13.728 SAR dan RMSE 30.517 SAR pada data testing — rata-rata prediksi meleset ±13.728 SAR dari harga sebenarnya, dengan R² 0,823 (model menjelaskan ~82,3% variasi harga). Faktor yang paling memengaruhi harga adalah `Year` (usia mobil), `Make`, `Mileage`, dan `Region`. Performa menurun di dua ujung segmen harga: mobil termurah (<30rb SAR, error persentase besar) dan termahal (>200rb SAR, data paling sedikit).

### 9.2 Actionable Recommendation

- Untuk mobil segmen mewah/harga tinggi, prediksi model sebaiknya dianggap sebagai estimasi minimum (floor), bukan harga pasti, karena kecenderungan underprediction.
- Penambahan data mobil terbaru (di luar snapshot 2021) akan meningkatkan relevansi model terhadap kondisi pasar saat ini.
- Aplikasi Streamlit dapat digunakan sebagai alat bantu awal bagi penjual/pembeli untuk estimasi harga wajar sebelum negosiasi lebih lanjut.

## 10. Limitations

1. **Performa menurun di segmen harga ekstrem** — MAPE tertinggi terjadi di mobil termurah (<30rb SAR, error persentase besar meski MAE absolut kecil) dan mobil termahal (>200rb SAR, data paling sedikit sehingga model kurang punya contoh untuk belajar pola harga di segmen ini).
2. **Objective tuning (MAE skala log) tidak sepenuhnya identik dengan business metric (MAE skala Riyal)** — `RandomizedSearchCV` memilih hyperparameter terbaik berdasarkan MAE skala log, sedangkan laporan akhir memakai MAE skala Riyal.
3. **Fitur terbatas pada atribut yang tercantum di listing** — tidak ada kondisi fisik detail, riwayat servis, atau riwayat kecelakaan, sehingga dua mobil dengan spesifikasi identik di data bisa saja punya harga wajar berbeda di dunia nyata.
4. **Model dilatih pada snapshot data 2021** — perlu retraining berkala mengikuti pergerakan harga pasar mobil bekas dari waktu ke waktu.

Catatan: listing dengan status "Negotiable" (`Price = 0`) dikeluarkan dari data training/testing, sehingga performa model di atas murni dari listing yang sudah berharga.

## 11. Deployment

Model dideploy sebagai aplikasi web interaktif menggunakan Streamlit:

🔗 https://deploymentprojectusedcar-knlk92haftjqjyqdaq56ts.streamlit.app/

Detail struktur & cara menjalankan aplikasi ada di [streamlit/README.md ](streamlit/README.md).

## 12. Dashboard

Selain aplikasi prediksi, proyek ini juga dilengkapi dengan dashboard interaktif untuk eksplorasi dataset menggunakan Tableau.

🔗 **Tableau Public:** [Used Car Intelligence Dashboard](https://public.tableau.com/app/profile/firsa.adam/viz/syarah_com/Dashboard1)

## 13. Contact

- Name   : Athiyyah Nisrina Husna 
- GitHub : [athiyyahnh99](https://github.com/athiyyahnh99)
- Email  : athiyyah.nh9@gmail.com
  
- Name   : Firsa Adam
- GitHub : [Firsaadam03](https://github.com/Firsaadam03)
- Email  : firsa00adam@gmail.com
