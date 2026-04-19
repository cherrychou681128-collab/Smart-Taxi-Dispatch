# Smart-Taxi-Dispatch: End-to-End MLOps & Full-Stack Architecture

## 📌 Project Overview
This project is a smart taxi dispatch system built for New York City. Instead of having drivers guess where their next passenger will be, this system uses AI to guide them to the right place at the right time.

We use two main models to make this happen:
* **ConvLSTM**: This model captures complex **spatial-temporal patterns** in past data, helping us exactly predict *when and where* the highest ride demand will happen.
* **XGBoost**: This looks at city data (like 311 service requests) to figure out how "valuable" or "profitable" a specific area is for drivers.

The ultimate goal is to send drivers to busy areas *before* passengers even request a ride. This cuts down wait times for passengers and helps drivers earn money more efficiently.

## 💾 Dataset & Large Files Download
Due to GitHub's strict file size limit (100MB), the massive deep learning tensors, complete NYC road network files, and the full datasets are not directly included in this repository. 

To run the full simulation, model training, and web app, please follow these steps:
1. Download the large files from this link: **[Google Drive Link](https://drive.google.com/drive/folders/1LJQ7n9PiDCiJlUS_GWxFO8NUqzvf2j8b?usp=sharing)**
2. Extract and place the files into their respective directories as follows:
   * **`data/` folder:** Place the deep learning tensors (`train_t24.npz`, etc.), Parquet datasets (`*_hourly.parquet`), and GIS Shapefiles (`taxi_zones.*`).
   * **`models/` folder:** Place the PyTorch weights `best.pt`.
   * **`simulation/` folder:** Place the full NYC network file `nyc.net.xml`.

*(Note: Lightweight sample files are provided in the repository for quick testing.)*

## 🚀 How to Run

### Option A: One-Click Automated AI Pipeline (MLOps)
This project includes a fully automated PowerShell script that handles feature engineering, model prediction, and simulation data generation.

```powershell
# Install all required Python dependencies
pip install -r requirements_full.txt

# Run the end-to-end pipeline
.\run_all.ps1
```
### Option B: Run the Full-Stack Web App
Start the interactive UI to see the dispatcher and driver maps in action.

```bash
# 1. Start the Node.js Backend Server
cd backend
npm install
node server.js

# 2. Start the React/Vite Frontend (Open a new terminal)
cd frontend
npm install
npm run dev
```
## 📂 Project Structure
* **`backend/`**: Node.js/Express server handling API requests.
* **`frontend/`**: Modern React application (Vite) featuring driver/passenger views and interactive Leaflet maps.
* **`train/`**: MLOps pipeline scripts for data preprocessing and ConvLSTM/Transformer model training.
* **`models/`**: AI engine housing the hybrid model logic and configuration files.
* **`simulation/`**: SUMO configuration files and scripts linking AI predictions to microscopic traffic routing.
* **`tools/`**: Utility scripts for data formatting and geographic coordinate conversions.
* **`data/`**: GIS boundaries, Parquet datasets, and spatial-temporal tensors.
