"""
Loads all trained model artifacts from Hugging Face Hub at startup.
Models are cached locally by huggingface_hub after the first download,
so subsequent restarts on the same machine are fast.
"""
import json
import joblib
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

HF_REPO_ID = "vikalp090/malaria-iaai"   # <-- set this to your real repo


class Time2Vec(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(1, 1)
        self.periodic = nn.Linear(1, dim - 1)

    def forward(self, t):
        t = t.unsqueeze(-1)
        return torch.cat([self.linear(t), torch.sin(self.periodic(t))], dim=-1)


class TransformerForecaster(nn.Module):
    def __init__(self, feat_dim, time_dim=8, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.time2vec = Time2Vec(time_dim)
        self.input_proj = nn.Linear(feat_dim + time_dim + 1, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=128, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.Linear(d_model, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid()
        )

    def forward(self, feats, times, values, is_query, key_padding_mask):
        t_emb = self.time2vec(times)
        val_masked = values * (1 - is_query)
        x = torch.cat([feats, t_emb, val_masked.unsqueeze(-1)], dim=-1)
        x = self.input_proj(x)
        out = self.encoder(x, src_key_padding_mask=key_padding_mask)
        return self.head(out).squeeze(-1)


class ModelStore:
    """Singleton-style holder for every artifact the backend needs."""

    def __init__(self):
        self.xgb_model = None
        self.lgb_model = None
        self.transformer = None
        self.feature_scaler = None
        self.scaler_geo = None
        self.scaler_clim = None
        self.regions_df = None
        self.neighbor_features = None
        self.region_history = None
        self.calibration = None
        self.country_density = None
        self.loaded = False

    def load(self):
        if self.loaded:
            return

        def dl(fname):
            return hf_hub_download(repo_id=HF_REPO_ID, filename=fname)

        self.xgb_model = joblib.load(dl("xgboost_final.joblib"))
        self.lgb_model = joblib.load(dl("lightgbm_final.joblib"))
        self.feature_scaler = joblib.load(dl("feature_scaler.joblib"))
        self.scaler_geo = joblib.load(dl("scaler_geo.joblib"))
        self.scaler_clim = joblib.load(dl("scaler_clim.joblib"))

        self.transformer = TransformerForecaster(feat_dim=8)
        self.transformer.load_state_dict(torch.load(dl("transformer_final.pt"), map_location="cpu"))
        self.transformer.eval()

        self.regions_df = pd.read_json(dl("regions_metadata.json"))
        self.neighbor_features = pd.read_json(dl("neighbor_features.json"))
        self.region_history = pd.read_json(dl("region_history.json"))
        self.country_density = pd.read_json(dl("country_density.json"))

        with open(dl("calibration.json")) as f:
            self.calibration = json.load(f)

        self.loaded = True
        print(f"ModelStore loaded: {len(self.regions_df)} regions, "
              f"{self.country_density['country'].nunique()} countries")


model_store = ModelStore()
