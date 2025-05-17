from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from pathlib import Path

app = FastAPI()

# Load model
model_path = Path(__file__).resolve().parent.parent / "model" / "fuel_model.pkl"
model = joblib.load(model_path)

class VehicleData(BaseModel):
    range_km: float
    mass_kg: float
    co2_emission_tons: float
    co2_wltp_g_per_km: float
    fuel_type: str
    fuel_mix: str
    engine_capacity_cc: float
    engine_power_kw: float
    energy_consumption_whpkm: float
    co2_reduction_wltp_gpkm: float
    electric_range_km: float

@app.post("/predict")
def predict(data: VehicleData):
    input_df = pd.DataFrame([data.dict()])
    pred = model.predict(input_df)[0]
    return {"fuel_consumption": round(pred, 2)}
