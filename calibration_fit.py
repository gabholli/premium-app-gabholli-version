import json, joblib, numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression

MODEL_PATH = "model.pkl"
SCALER_PATH = "scaler.pkl"
CALIB_CSV = "calib_samples.csv"
CALIB_JSON = "calib.json"

df = pd.read_csv(CALIB_CSV)
features = ["Bedrooms", "Square Footage", "Coverage A", "Age of Home"]
X = df[features].astype(float).values
y_true = df["ActualPremium"].astype(float).values

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

X_scaled = scaler.transform(X)
y_pred = model.predict(X_scaled)

reg = LinearRegression().fit(y_pred.reshape(-1, 1), y_true)
a = float(reg.coef_[0]); b = float(reg.intercept_)

# small shrinkage for stability with only 3 points
alpha = 0.3
a = (1 - alpha) * a + alpha * 1.0
b = (1 - alpha) * b + alpha * 0.0

with open(CALIB_JSON, "w") as f:
    json.dump({"a": a, "b": b}, f)

print({"a": a, "b": b})