# Smart-Taxi-Dispatch
# Intelligent Taxi Dispatch System

## 📌 Project Overview
This project presents an intelligent taxi dispatch and routing system designed to optimize fleet management in New York City. By leveraging a hybrid machine learning approach, the system combines **ConvLSTM** for capturing complex spatial-temporal patterns in ride-hailing demand and **XGBoost** for evaluating localized reward scores based on urban features (e.g., 311 service requests). The ultimate goal is to proactively guide drivers to high-demand areas, reducing passenger wait times and maximizing driver efficiency.

## 💾 Dataset & Large Files Download
Due to GitHub's strict file size limit (100MB), the complete NYC road network file and the full 311 dataset are not directly included in this repository. 

To run the full simulation and model training, please follow these steps:
1. Download the large files from this link: **[Insert Your Google Drive/Cloud Link Here]**
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
