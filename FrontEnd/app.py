# app.py
from flask import Flask, render_template, request
import numpy as np
import os
import json

# Try imports that may or may not be installed in your env
try:
    import joblib
except Exception:
    joblib = None

try:
    import torch
    from torch import nn
except Exception:
    torch = None
    nn = None

app = Flask(__name__)

# -------------------------------------------------------------------
# Load labels list (extracted from your weather.csv). If not present,
# a small default list is used.
# -------------------------------------------------------------------
LABELS_PATHS = [
    "weather_labels.json",                             # prefer this in app folder
    os.path.join("..", "Project_extracted", "Project", "weather_labels.json")  # if you keep provided structure
]

labels = None
for p in LABELS_PATHS:
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f:
            labels = json.load(f)
        break

if labels is None:
    # fallback short list (will still work)
    labels = ["Clear", "Cloudy", "Rainy", "Foggy"]

# -------------------------------------------------------------------
# Helper: normalize/prepare user features into numpy array
# -------------------------------------------------------------------
def parse_features(form):
    vals = [
        float(form["Temp_C"]),
        float(form["Dew_Point_Temp_C"]),
        float(form["Rel_Hum"]),
        float(form["Wind_Speed"]),
        float(form["Visibility"]),
        float(form["Pressure"])
    ]
    return np.array([vals], dtype=np.float32)

# -------------------------------------------------------------------
# Attempt to load model using several strategies.
# The loader returns a callable `predict_callable(X)` that takes numpy X
# and returns a label string (or first label if multiple).
# -------------------------------------------------------------------
def get_model_predictor():
    # 1) Try joblib.load and look for predict()
    if joblib is not None and os.path.exists("model.joblib"):
        try:
            mdl = joblib.load("model.joblib")
            # If it has a scikit-like predict method, use it
            if hasattr(mdl, "predict"):
                def predict_callable(X_np):
                    out = mdl.predict(X_np)
                    # out might be array of strings or numeric labels
                    if isinstance(out, (list, np.ndarray)):
                        val = out[0]
                    else:
                        val = out
                    # if numeric, map via labels (safe)
                    try:
                        if isinstance(val, (int, np.integer)):
                            return labels[int(val)]
                    except Exception:
                        pass
                    # otherwise convert to string
                    return str(val)
                return predict_callable
        except Exception as e:
            print("joblib load failed or object not scikit-like:", e)

    # 2) Try torch.load of a full PyTorch object (if present)
    if torch is not None and os.path.exists("model.joblib"):
        try:
            # some people saved with torch.save(model, "model.joblib")
            mdl = torch.load("model.joblib", map_location="cpu")
            if isinstance(mdl, torch.nn.Module):
                mdl.eval()
                # If the loaded module has a direct `predict` or returns label indexes/strings:
                def predict_callable(X_np):
                    with torch.no_grad():
                        x = torch.tensor(X_np, dtype=torch.float32)
                        try:
                            out = mdl(x)
                        except Exception as ex:
                            # If module implements a .predict method
                            if hasattr(mdl, "predict"):
                                out = mdl.predict(X_np)
                                # ensure we return string
                                if isinstance(out, (list, np.ndarray)):
                                    return str(out[0])
                                return str(out)
                            raise ex

                        # out might be logits (tensor), or probabilities, or direct labels
                        if isinstance(out, torch.Tensor):
                            # if dtype is float, get argmax
                            if out.dim() == 2:
                                idx = int(torch.argmax(out, dim=1).item())
                                # Ensure idx in labels bounds
                                if 0 <= idx < len(labels):
                                    return labels[idx]
                                else:
                                    return str(idx)
                            else:
                                return str(out.numpy().tolist())
                        else:
                            return str(out)
                return predict_callable
        except Exception as e:
            print("torch.load(model.joblib) failed:", e)

    # 3) Try loading state_dict (model_weights.pth) into a generic MLP
    #    This is a best-effort fallback. If your real model has different
    #    architecture this will likely be inaccurate. Prefer to export
    #    state_dict from the training notebook using torch.save(model.state_dict()).
    if torch is not None and os.path.exists("model_weights.pth") and nn is not None:
        try:
            class GenericMLP(nn.Module):
                def __init__(self, in_features=6, hidden=64, out_features=len(labels)):
                    super().__init__()
                    self.net = nn.Sequential(
                        nn.Linear(in_features, hidden),
                        nn.ReLU(),
                        nn.Linear(hidden, max(16, hidden//2)),
                        nn.ReLU(),
                        nn.Linear(max(16, hidden//2), out_features)
                    )
                def forward(self, x):
                    return self.net(x)

            mdl = GenericMLP()
            mdl.load_state_dict(torch.load("model_weights.pth", map_location="cpu"))
            mdl.eval()
            def predict_callable(X_np):
                x = torch.tensor(X_np, dtype=torch.float32)
                with torch.no_grad():
                    out = mdl(x)
                    idx = int(torch.argmax(out, dim=1).item())
                    return labels[idx] if 0 <= idx < len(labels) else str(idx)
            return predict_callable
        except Exception as e:
            print("loading state_dict into GenericMLP failed:", e)

    # 4) Last resort: a deterministic dummy predictor so the app never crashes
    def dummy_predict(X_np):
        # simple deterministic rule: use visibility + humidity heuristics
        x = X_np[0]
        temp, dew, hum, wind, vis, pres = x
        if vis < 1.5 or hum > 90:
            return "Fog"
        if hum > 80 and temp > 0 and vis > 2:
            return "Rain"
        if hum < 40 and temp > 20:
            return "Clear"
        return "Cloudy"
    return dummy_predict

# Create the predictor callable once (on start)
predictor = get_model_predictor()

# -------------------------------------------------------------------
# Flask routes
# -------------------------------------------------------------------
@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict_route():
    try:
        X = parse_features(request.form)
        label = predictor(X)
        return render_template("index.html", result=f"Predicted Weather: {label}")
    except Exception as e:
        return render_template("index.html", result=f"Error: {str(e)}")

if __name__ == "__main__":
    # recommended: set host and debug depending on env
    app.run(debug=True)
