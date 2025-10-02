# CapstoneProject2Purwadhika

# 🏠 Prediksi Harga Apartemen di Daegu dengan Machine Learning

Proyek ini bertujuan untuk memprediksi **harga jual apartemen di Daegu, Korea Selatan** dengan menggunakan **machine learning**.  
Model dikembangkan di **Google Colab** dan disimpan dalam bentuk file `.sav` sehingga dapat digunakan kembali untuk prediksi pada data baru.

---

## 📌 Dataset
Dataset berisi informasi apartemen di Daegu, seperti:
- **HallwayType** (jenis koridor: corridor, mixed, terraced)  
- **TimeToSubway** (estimasi waktu ke stasiun subway terdekat: 0-5min, 5-10min, 10-15min, 15+min, no_subway_nearby)  
- **Jumlah fasilitas terdekat** (sekolah, kantor publik, universitas, parkiran basement, dsb)  
- **Ukuran apartemen (sqf)**  
- **SalePrice** (harga jual dalam Won)

---

## ⚙️ Model yang Digunakan
- **XGBoost Regressor (XGBRegressor)**  
- Pipeline preprocessing:
  - `OneHotEncoder` → untuk `HallwayType`  
  - `OrdinalEncoder` → untuk `TimeToSubway` (urutan waktu dekat → jauh)  
  - Fitur numerik lainnya dipertahankan (`passthrough`)  

Model dilatih dengan `Pipeline` dan disimpan menggunakan `pickle`.

---

## 📊 Evaluasi Model
Model dievaluasi menggunakan 3 metrik:
- **RMSE (Root Mean Squared Error)** → mengukur rata-rata selisih kuadrat prediksi dan nilai aktual.  
- **MAE (Mean Absolute Error)** → mengukur rata-rata kesalahan absolut.  
- **MAPE (Mean Absolute Percentage Error)** → mengukur kesalahan dalam bentuk persentase.

Semakin rendah nilai ketiga metrik ini, semakin baik performa model.

---

## 💾 Menyimpan & Memuat Model
### Simpan model:
```python
import pickle

with open('Model_DaeguApartment_XGB.sav', 'wb') as f:
    pickle.dump(pipeline, f)
