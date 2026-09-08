# Malaria Forecast Assistant

A RAG-grounded forecasting assistant for malaria prevalence across six
sub-Saharan African countries. Routes each query to whichever model
(XGBoost, LightGBM, or Transformer) was found most reliable for that
region's data density, returns a calibrated 90% interval rather than a
raw model output, and explains predictions with real per-prediction SHAP
feature contributions plus a Groq-powered LLM that shows its reasoning
explicitly (Estimate / Reasoning / Caveats), grounded in the project's
own findings and PubMed literature.

## Before running anything

1. Open `backend/models.py` and set `HF_REPO_ID` to your real Hugging Face
   repo, e.g. `"yourusername/malaria-iaai"`.
2. Get a free Groq API key at https://console.groq.com and set it as the
   `GROQ_API_KEY` environment variable. Without it, the app still runs,
   `/ask` falls back to a plain deterministic template instead of an
   LLM-composed, reasoning-based answer.

## Run locally

```bash
cd backend
pip install -r requirements.txt
uvicorn app:app --reload
```

Then open `http://localhost:8000` in a browser.

## Deploy (Render, free tier)

1. Push this repo to GitHub.
2. On [render.com](https://render.com), create a **New Web Service**,
   connect your GitHub repo.
3. Set **Root Directory** to `backend`.
4. Set **Build Command** to `pip install -r requirements.txt`.
5. Set **Start Command** to `uvicorn app:app --host 0.0.0.0 --port $PORT`.
6. Add environment variable `GROQ_API_KEY` (recommended) under Environment
   Variables.
7. Deploy. Render gives you a public URL, e.g. `https://malaria-iaai.onrender.com`.

The first request after a period of inactivity may be slow on Render's
free tier (the service spins down and cold-starts), that is expected.

## Project structure

```
backend/
  app.py            FastAPI app: routing, forecast, compare, ask, overview, findings
  models.py          Loads trained models from Hugging Face Hub
  routing.py          Density-based routing logic
  inference.py         Builds features, runs the routed model, calibrates the
                         interval, computes SHAP explanations (XGBoost/LightGBM)
  rag.py                TF-IDF retrieval over knowledge/ (+ optional pubmed_abstracts.json)
  query_parser.py        Extracts country/region/intent from natural-language queries
  knowledge/               Project findings, used to ground /ask responses
frontend/
  index.html                Tabs: Estimate (map + search + auto-narrative),
                              All Countries (overview grid), Compare Models,
                              Ask, Methodology
```

## Endpoints

- `GET  /regions` -- all countries and their regions
- `GET  /history/{region_id}` -- historical prevalence series for charting
- `GET  /overview` -- trend + current estimate for all six countries at once
- `GET  /findings` -- project findings documents, for the Methodology tab
- `POST /forecast` -- routed (or manually overridden) prediction for one region
- `POST /compare` -- runs all three models on the same region side by side
- `POST /ask` -- natural-language query, LLM-composed answer with explicit reasoning

## Known limitations (stated honestly, not hidden)

- Trained on survey data through 2015 only. There is no live data feed, so
  this reflects historical status, not a live forecast into the current year.
- The Graph Neural ODE architecture, evaluated during this project, is not
  used to serve predictions, as it was consistently the least reliable
  model tested. It is documented as a negative result, not deployed.
- Per-prediction SHAP explanations are only available for XGBoost and
  LightGBM; the Transformer has no fast SHAP explainer wired up here, and
  the assistant is instructed to say so plainly if asked.
- Uncertainty intervals are calibrated against historical residuals from
  evaluation; they are not a guarantee for genuinely novel conditions.
