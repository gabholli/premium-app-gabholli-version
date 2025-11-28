from __future__ import annotations
from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import os
import json

# -------------------------
# App setup
# -------------------------
app = Flask(__name__, template_folder="templates", static_folder="static")

# Artifact paths (env overrides allowed)
MODEL_PATH = os.getenv("MODEL_PATH", "model.pkl")
SCALER_PATH = os.getenv("SCALER_PATH", "scaler.pkl")
CALIB_PATH = os.getenv("CALIB_PATH", "calib.json")

# Lazy-loaded artifacts
_model = None
_scaler = None
_calib = None

def load_artifacts():
    """Load model & scaler once."""
    global _model, _scaler
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    if _scaler is None:
        _scaler = joblib.load(SCALER_PATH)

def load_calibration():
    """Load optional calibration {a, b} once."""
    global _calib
    if _calib is None and os.path.exists(CALIB_PATH):
        with open(CALIB_PATH, "r") as f:
            _calib = json.load(f)

# -------------------------
# Routes
# -------------------------
@app.route("/health")
def health():
    return {"status": "ok"}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    load_artifacts()
    load_calibration()

    # Accept JSON or form-encoded submissions
    data = request.get_json(silent=True) or request.form.to_dict()

    # Prefer Coverage A; fall back to Property Value if provided
    covA = data.get("Coverage A", data.get("Property Value"))

    required_base = ["Bedrooms", "Square Footage", "Age of Home"]
    missing = [k for k in required_base if k not in data]
    if covA is None:
        missing.append("Coverage A (or Property Value)")

    if missing:
        return jsonify({"error": f"Missing fields: {', '.join(missing)}"}), 400

    # Parse numbers safely
    try:
        bedrooms = float(data["Bedrooms"])
        sqft = float(data["Square Footage"])
        coverage_a = float(covA)
        age = float(data["Age of Home"])
    except Exception as e:
        return jsonify({"error": f"Invalid input types: {e}"}), 400

    # Feature order must match training
    X = np.array([bedrooms, sqft, coverage_a, age], dtype=float).reshape(1, -1)

    # Scale + predict
    Xs = _scaler.transform(X)
    pred = float(_model.predict(Xs)[0])

    # Optional calibration: y_cal = a*y + b
    pred_cal = pred
    if _calib and "a" in _calib and "b" in _calib:
        pred_cal = float(_calib["a"]) * pred + float(_calib["b"])
        pred_cal = max(0.0, pred_cal)  # keep sane

    return jsonify({
        "predicted_premium": round(pred_cal, 2)
        # uncomment to debug:
        # "base_prediction": round(pred, 2),
        # "calibration": _calib,
    })

if __name__ == "__main__":
    # For local dev
    app.run(host="0.0.0.0", port=5000, debug=True)
