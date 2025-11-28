# =============================
# train_and_serialize.py
# =============================
"""
Trains a simple linear regression model on your simulated dataset and writes:
  - model.pkl
  - scaler.pkl

Usage:
  python train_and_serialize.py --csv simulated_home_insurance_quotes.csv
"""
import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def main(csv_path: str, out_model: str = "model.pkl", out_scaler: str = "scaler.pkl"):
    df = pd.read_csv(csv_path)

    # Defensive drops if present
    for col in ["ZIP Code", "Year"]:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    # Create Age of Home from Year Built (if present)
    if "Year Built" in df.columns and "Age of Home" not in df.columns:
        df["Age of Home"] = 2025 - df["Year Built"]
        df.drop(columns=["Year Built"], inplace=True)

    # Convert common Yes/No columns to 0/1 if present (safe no-ops otherwise)
    yn_cols = [
        "Has Swimming Pool",
        "Security System Installed",
        "Has Garage",
        "Has Basement",
    ]
    for c in yn_cols:
        if c in df.columns and df[c].dtype == object:
            df[c] = (df[c].str.strip().str.lower() == "yes").astype(int)

    # Ensure Coverage A exists (fallback to Property Value for backward compatibility)
    if "Coverage A" not in df.columns:
        if "Coverage A" in df.columns:
            df["Coverage A"] = df["Coverage A"]
        else:
            raise ValueError("CSV must include 'Coverage A' (or 'Property Value' as a fallback).")

    # Select features used in your notebook
    features = ["Bedrooms", "Square Footage", "Coverage A", "Age of Home"]
    missing_feats = [c for c in features if c not in df.columns]
    if missing_feats:
        raise ValueError(f"CSV is missing required columns: {missing_feats}")

    X = df[features].copy()
    y = df["Premiums Per Policy"].astype(float).copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_s, y_train)

    # Eval
    pred = model.predict(X_test_s)
    mae = mean_absolute_error(y_test, pred)
    rmse = mean_squared_error(y_test, pred, squared=False)
    r2 = r2_score(y_test, pred)
    print({"MAE": mae, "RMSE": rmse, "R2": r2})

    # Persist artifacts
    joblib.dump(model, out_model)
    joblib.dump(scaler, out_scaler)
    print(f"Saved: {out_model}, {out_scaler}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="simulated_home_insurance_quotes.csv")
    ap.add_argument("--model", default="model.pkl")
    ap.add_argument("--scaler", default="scaler.pkl")
    args = ap.parse_args()
    main(args.csv, args.model, args.scaler)