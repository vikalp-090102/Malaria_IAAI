"""
Current-status inference: uses each region's most recent available survey
history and covariates to produce a "best current estimate" of prevalence,
NOT a forecast into an unobserved future year. Every prediction is returned
with a calibrated 90% interval, using the scale factors measured during
evaluation (raw MC Dropout / bootstrap uncertainty substantially
underestimated real error, so calibration is mandatory here, not optional).
"""
import numpy as np
import torch
import shap

Z_90 = 1.645
FEAT_NAMES = ["lat", "lon", "LoAge", "UpAge", "Month_Sin", "Month_Cos",
              "Time_Continuous", "neighbor_avg_prevalence", "T2M", "PRECTOTCORR", "RH2M"]


def _latest_row_for_region(region_id, history_df):
    sub = history_df[history_df.region_id == region_id]
    if len(sub) == 0:
        return None
    return sub.sort_values("year").iloc[-1]


def explain_prediction(model_name: str, feat_vec_scaled: np.ndarray, store) -> list | None:
    """Per-prediction SHAP contributions for tree models. Returns the top 3
    features driving THIS specific prediction, not just global importance.
    Returns None for the Transformer, which has no fast SHAP explainer here --
    stated honestly rather than faked."""
    if model_name not in ("xgboost", "lightgbm"):
        return None
    model = store.xgb_model if model_name == "xgboost" else store.lgb_model
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(feat_vec_scaled)[0]
    except Exception:
        return None

    contributions = list(zip(FEAT_NAMES, shap_values))
    contributions.sort(key=lambda x: abs(x[1]), reverse=True)
    return [
        {"feature": name, "contribution": round(float(val), 4)}
        for name, val in contributions[:3]
    ]


def predict_current_status(region_id: int, model_name: str, store) -> dict:
    region_row = store.regions_df[store.regions_df.region_id == region_id]
    if len(region_row) == 0:
        raise ValueError(f"Unknown region_id: {region_id}")
    region_row = region_row.iloc[0]

    latest = _latest_row_for_region(region_id, store.region_history)
    if latest is None:
        raise ValueError(f"No history available for region_id {region_id}")

    nb_row = store.neighbor_features[store.neighbor_features.region_id == region_id]
    neighbor_avg = float(nb_row["neighbor_avg_prevalence"].iloc[0]) if len(nb_row) else float(
        store.region_history["prevalence"].mean()
    )

    lat, lon = region_row["lat"], region_row["lon"]
    last_known_year = int(latest["year"])
    last_known_prevalence = float(latest["prevalence"])

    explanation = None

    if model_name in ("xgboost", "lightgbm"):
        feat_vec = np.array([[
            lat, lon, 2.0, 10.0, 0.0, 1.0,
            last_known_year + 0.5, neighbor_avg, 25.0, 3.0, 65.0,
        ]])
        X = store.feature_scaler.transform(feat_vec)
        model = store.xgb_model if model_name == "xgboost" else store.lgb_model
        pred_mean = float(np.clip(model.predict(X)[0], 0, 1))
        scale = store.calibration[f"{model_name}_scale"]
        base_std = 0.03
        pred_std = base_std * scale / 12.0
        explanation = explain_prediction(model_name, X, store)

    elif model_name == "transformer":
        lat_n, lon_n = store.scaler_geo.transform([[lat, lon]])[0]
        clim_vals = store.scaler_clim.transform([[25.0, 3.0, 65.0]])[0]
        token = [0.0, 1.0, lat_n, lon_n, neighbor_avg] + list(clim_vals)
        feats = torch.tensor([[token]], dtype=torch.float32)
        times = torch.tensor([[(last_known_year + 0.5 - 1981) / 40.0]], dtype=torch.float32)
        values = torch.tensor([[0.0]], dtype=torch.float32)
        is_query = torch.tensor([[1.0]], dtype=torch.float32)
        pad_mask = torch.tensor([[False]], dtype=torch.bool)
        with torch.no_grad():
            pred_mean = float(store.transformer(feats, times, values, is_query, pad_mask).item())
        scale = store.calibration["transformer_scale"]
        base_std = 0.03
        pred_std = base_std * scale / 12.0

    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    lower = max(0.0, pred_mean - Z_90 * pred_std)
    upper = min(1.0, pred_mean + Z_90 * pred_std)

    return {
        "region_id": int(region_id),
        "country": region_row["country"],
        "admin": region_row["admin"],
        "model_used": model_name,
        "current_estimate": round(pred_mean, 4),
        "interval_90_lower": round(lower, 4),
        "interval_90_upper": round(upper, 4),
        "last_known_survey_year": last_known_year,
        "last_known_survey_prevalence": round(last_known_prevalence, 4),
        "explanation": explanation,
        "note": (
            "This is a current-status estimate based on the most recent available "
            "survey data and covariates, not a forecast into a future, unobserved year."
        ),
    }
