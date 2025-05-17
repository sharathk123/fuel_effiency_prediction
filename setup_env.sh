#!/bin/bash

# Set the virtual environment name
ENV_NAME="fuel_efficiency_env"

# Create the virtual environment
python3 -m venv $ENV_NAME

# Activate the virtual environment
source $ENV_NAME/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

echo "✅ Virtual environment '$ENV_NAME' setup complete and activated."

