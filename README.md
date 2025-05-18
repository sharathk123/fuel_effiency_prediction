# 🚗 Fuel Efficiency Prediction

This project aims to build a machine learning model that predicts vehicle fuel consumption (in L/100km) using various technical and environmental attributes. It supports both **batch prediction** and **real-time inference** through a FastAPI service.

---

## ⚙️ How to Run the FastAPI Server

### 🔹 Set Up Environment and Install Dependencies

```bash
python3 -m venv fuel_efficiency_env
source fuel_efficiency_env/bin/activate
pip install -r requirements.txt
```

### 🔹 Train Model and Start the API Server

```bash
python model/train_model.py
cd api
uvicorn main:app --reload
```

API will be available at: [http://127.0.0.1:8000](http://127.0.0.1:8000)

---

## 📫 Sample JSON for `/predict`

```json
{
  "range_km": 1,
  "mass_kg": 1450,
  "co2_emission_tons": 0.125,
  "co2_wltp_g_per_km": 135,
  "engine_capacity_cc": 1400,
  "engine_power_kw": 85,
  "energy_consumption_whpkm": 180,
  "co2_reduction_wltp_gpkm": 5,
  "electric_range_km": 0,
  "fuel_type": "petrol",
  "fuel_mix": "M"
}
```
