# predict_model.py
import pandas as pd
import joblib

# Load the model
model = joblib.load("model/fuel_model.pkl")

# Load production data (last 100k rows)
df = pd.read_csv("data/Automobile_data.csv", sep=",")
df.columns = df.columns.str.strip()

# Drop unnamed column if it exists
if df.columns[0].lower().startswith("unnamed"):
    df.drop(columns=df.columns[0], inplace=True)

# Rename columns
df.rename(columns={
    'r': 'range_km',
    'm (kg)': 'mass_kg',
    'Mt': 'co2_emission_tons',
    'Ewltp (g/km)': 'co2_wltp_g_per_km',
    'Ft': 'fuel_type',
    'Fm': 'fuel_mix',
    'ec (cm3)': 'engine_capacity_cc',
    'ep (KW)': 'engine_power_kw',
    'z (Wh/km)': 'energy_consumption_whpkm',
    'Erwltp (g/km)': 'co2_reduction_wltp_gpkm',
    'Fuel consumption': 'fuel_consumption',
    'Electric range (km)': 'electric_range_km'
}, inplace=True)

# Keep only the last 100k for production prediction
production_data = df.tail(100000).copy()

# Drop rows with missing target (if any)
production_data = production_data[production_data['fuel_consumption'].notna()]

# Separate features and actual target (if available)
X_prod = production_data.drop("fuel_consumption", axis=1)
y_true = production_data["fuel_consumption"]

# Predict
predictions = model.predict(X_prod)

# Combine with actuals for inspection
results = X_prod.copy()
results["actual"] = y_true
results["predicted"] = predictions

# Show sample results
print(results[["actual", "predicted"]].head())
