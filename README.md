# Smart-Taxi-Dispatch
# Intelligent Taxi Dispatch System

## 📌 Project Overview
This project is a smart taxi dispatch system built for New York City. Instead of having drivers guess where their next passenger will be, this system uses AI to guide them to the right place at the right time.

We use two main models to make this happen:

1.ConvLSTM: This model captures complex spatial-temporal patterns in past data, helping us exactly predict when and where the highest ride demand will happen.

2.XGBoost: This looks at city data (like 311 service requests) to figure out how "valuable" or "profitable" a specific area is for drivers.

The ultimate goal is send drivers to busy areas before passengers even request a ride. This cuts down wait times for passengers and helps drivers earn money more efficiently.

## 💾 Dataset & Large Files Download
Due to GitHub's strict file size limit (100MB), the complete NYC road network file and the full 311 dataset are not directly included in this repository. 

To run the full simulation and model training, please follow these steps:
1. Download the large files from this link: **[(https://drive.google.com/drive/folders/1LJQ7n9PiDCiJlUS_GWxFO8NUqzvf2j8b?usp=sharing)]**
2. Place the network file `nyc.net.xml` into the `simulation/` directory.
3. Place the raw dataset `nyc_311_2025_07.csv` into the `data/` directory.

*(Note: For quick testing, a lightweight `grid.net.xml` and a sample dataset are provided in the repository.)*

## 🚀 How to Run

### Step 1: Feature Engineering & Model Prediction
First, process the raw data and generate the demand predictions for the next hour.
```bash
# Calculate zone rewards based on 311 data
python models/build_zone_reward_from_311.py

# Run the hybrid model to predict and rank next-hour demand
python models/predict_next_hour_advanced.py
