# train_model.py
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import os

# Read the CSV correctly
df = pd.read_csv('data/Automobile_data.csv', sep=',')

# Strip leading/trailing spaces from column names
df.columns = df.columns.str.strip()

# Show original columns
print("🔎 Original columns:", df.columns.tolist())

# Drop the first column if it's unnamed or a serial index
if df.columns[0].lower().startswith("unnamed") or df.columns[0].strip() == "":
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

# Confirm renaming worked
print("✅ Renamed columns:", df.columns.tolist())

# Drop duplicates and rows with missing target
df.drop_duplicates(inplace=True)
df = df[df['fuel_consumption'].notna()]

# Dataset split
train_df = df.iloc[:700000]
val_df = df.iloc[700000:900000]
test_df = df.iloc[900000:]

# Features and target
X = train_df.drop(columns=['fuel_consumption'])
y = train_df['fuel_consumption']

# Identify numeric and categorical columns
num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = X.select_dtypes(include=['object']).columns.tolist()

# Preprocessing transformers
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Full preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, num_cols),
        ('cat', categorical_transformer, cat_cols)
    ]
)

# Final pipeline with model
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Train/test split (first 700k for training)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)

# Train the pipeline
print("🚀 Training model...")
pipeline.fit(X_train, y_train)

# Save the model
os.makedirs("model", exist_ok=True)
joblib.dump(pipeline, 'model/fuel_model.pkl')

val_df.to_csv('data/validation_data.csv', index=False)
test_df.to_csv('data/test_data.csv', index=False)
print("✅ Model trained and saved successfully.")