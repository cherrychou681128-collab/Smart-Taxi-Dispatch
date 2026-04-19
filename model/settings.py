import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(ROOT_DIR, "data")
MODEL_DIR = os.path.join(ROOT_DIR, "models")
FEATURE_DIR = os.path.join(ROOT_DIR, "features")

TRAIN_PARQUET = os.path.join(DATA_DIR, "train_hourly.parquet")
VALID_PARQUET = os.path.join(DATA_DIR, "valid_hourly.parquet")
TEST_PARQUET  = os.path.join(DATA_DIR, "test_hourly.parquet")

XGB_MODEL_PATH = os.path.join(MODEL_DIR, "xgb.model")

ZONE_COL = "PULocationID"
TIME_COL = "pickup_hour"
TARGET_COL = "rides"