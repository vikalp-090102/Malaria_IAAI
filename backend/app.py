"""
Main backend for the malaria forecasting assistant.

Endpoints:
  GET  /health              -- liveness check
  GET  /regions               -- list of countries and their regions
  GET  /history/{region_id}    -- historical prevalence series for charting
  GET  /findings                -- project findings documents, for the About tab
  POST /forecast                  -- structured request, optional model_override
  POST /ask                        -- natural language request, LLM-composed

Serves the static frontend from /frontend as well, so a single deploy
(one Render service) gives one public URL for everything.
"""
import os
import glob
import json
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from models import model_store
from routing import route
from inference import predict_current_status
from rag import rag_index
from query_parser import parse_query

app = FastAPI(title="Malaria Forecast Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
KNOWLEDGE_DIR = os.path.join(os.path.dirname(__file__), "knowledge")


@app.on_event("startup")
def startup():
    model_store.load()
    rag_index.build()
    if not GROQ_API_KEY:
        print("WARNING: GROQ_API_KEY not set. /ask will use a plain template "
              "response instead of LLM-composed answers. Set this environment "
              "variable on your host for full functionality (free at console.groq.com).")


# ----------------------------------------------------------------------
# Request schemas
# ----------------------------------------------------------------------
class ForecastRequest(BaseModel):
    country: str
    region_id: int | None = None
    model_override: str | None = None   # "xgboost" | "lightgbm" | "transformer"


class AskRequest(BaseModel):
    query: str
    register: str = "researcher"  # "researcher" | "health_officer"


# ----------------------------------------------------------------------
# Core data endpoints
# ----------------------------------------------------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "models_loaded": model_store.loaded,
        "rag_loaded": rag_index.loaded,
        "llm_enabled": bool(GROQ_API_KEY),
    }


@app.get("/regions")
def list_regions():
    df = model_store.regions_df
    out = {}
    for country, group in df.groupby("country"):
        out[country] = [
            {"region_id": int(r.region_id), "admin": r.admin, "lat": r.lat, "lon": r.lon}
            for r in group.itertuples()
        ]
    return out


@app.get("/history/{region_id}")
def region_history(region_id: int):
    hist = model_store.region_history[model_store.region_history.region_id == region_id]
    if len(hist) == 0:
        raise HTTPException(404, f"No history for region_id {region_id}")
    hist = hist.sort_values("year")
    return {
        "region_id": region_id,
        "years": hist["year"].tolist(),
        "prevalence": hist["prevalence"].tolist(),
    }


@app.get("/findings")
def findings():
    docs = []
    for path in sorted(glob.glob(os.path.join(KNOWLEDGE_DIR, "*.md"))):
        with open(path, encoding="utf-8") as f:
            text = f.read()
        title = os.path.basename(path).replace(".md", "").replace("_", " ").title()
        docs.append({"title": title, "text": text})
    return {"documents": docs}


def _density_lookup():
    d = model_store.country_density
    return dict(zip(d["country"], d["n_regions"]))


# ----------------------------------------------------------------------
# Forecast (supports model override, used by the Compare Models feature)
# ----------------------------------------------------------------------
VALID_MODELS = {"xgboost", "lightgbm", "transformer"}


@app.get("/overview")
def overview():
    """One trend + current estimate per country, for the all-countries view."""
    density = _density_lookup()
    results = []
    for country, group in model_store.regions_df.groupby("country"):
        region_ids = group["region_id"].tolist()
        hist = model_store.region_history[model_store.region_history.region_id.isin(region_ids)]
        trend = hist.groupby("year")["prevalence"].mean().reset_index().sort_values("year")

        # representative region = the one with the most historical observations
        counts = hist.groupby("region_id").size()
        rep_region_id = int(counts.idxmax()) if len(counts) else int(group.iloc[0].region_id)

        routing_decision = route(country, density)
        try:
            current = predict_current_status(rep_region_id, routing_decision["model"], model_store)
        except ValueError:
            current = None

        results.append({
            "country": country,
            "n_regions": len(region_ids),
            "years": trend["year"].tolist(),
            "prevalence": trend["prevalence"].tolist(),
            "current_estimate": current["current_estimate"] if current else None,
            "interval_90_lower": current["interval_90_lower"] if current else None,
            "interval_90_upper": current["interval_90_upper"] if current else None,
            "model_used": routing_decision["model"],
            "representative_region_id": rep_region_id,
        })
    return {"countries": results}


@app.post("/forecast")
def forecast(req: ForecastRequest):
    region_id = req.region_id
    if region_id is None:
        sub = model_store.regions_df[model_store.regions_df.country == req.country]
        if len(sub) == 0:
            raise HTTPException(404, f"Unknown country: {req.country}")
        region_id = int(sub.iloc[0].region_id)

    if req.model_override:
        if req.model_override not in VALID_MODELS:
            raise HTTPException(400, f"model_override must be one of {VALID_MODELS}")
        chosen_model = req.model_override
        routing_reason = f"Manually selected: {chosen_model} (bypassing automatic routing for comparison)."
    else:
        density = _density_lookup()
        routing_decision = route(req.country, density)
        chosen_model = routing_decision["model"]
        routing_reason = routing_decision["reason"]

    try:
        result = predict_current_status(region_id, chosen_model, model_store)
    except ValueError as e:
        raise HTTPException(400, str(e))

    result["routing_reason"] = routing_reason
    return result


@app.post("/compare")
def compare_models(req: ForecastRequest):
    """Runs all three models on the same region for direct side-by-side comparison."""
    region_id = req.region_id
    if region_id is None:
        sub = model_store.regions_df[model_store.regions_df.country == req.country]
        if len(sub) == 0:
            raise HTTPException(404, f"Unknown country: {req.country}")
        region_id = int(sub.iloc[0].region_id)

    density = _density_lookup()
    routing_decision = route(req.country, density)

    results = {}
    for m in ("xgboost", "lightgbm", "transformer"):
        try:
            r = predict_current_status(region_id, m, model_store)
            r["is_routed_choice"] = (m == routing_decision["model"])
            results[m] = r
        except ValueError as e:
            results[m] = {"error": str(e)}

    return {"region_id": region_id, "routed_model": routing_decision["model"],
            "routing_reason": routing_decision["reason"], "results": results}


# ----------------------------------------------------------------------
# LLM-primary answer composer
# ----------------------------------------------------------------------
LANGUAGE_INSTRUCTIONS = {
    "english": "Respond in English.",
}

REGISTER_INSTRUCTIONS = {
    "researcher": (
        "Write for a malaria research audience: precise, willing to mention model "
        "names, R-squared, calibration, and methodology where relevant."
    ),
    "health_officer": (
        "Write for a district health officer with no machine learning background: "
        "plain language, no jargon, focus on what the numbers mean for decisions, "
        "briefly explain any technical term you must use."
    ),
}


def _compose_answer(parsed: dict, forecast_result: dict | None, retrieved: list,
                     register: str) -> str:
    context_text = "\n\n".join(f"[{r['source']}] {r['text']}" for r in retrieved)
    forecast_text = json.dumps(forecast_result, indent=2) if forecast_result else "No specific region matched."

    explanation_text = ""
    if forecast_result and forecast_result.get("explanation"):
        explanation_text = (
            "\n\nModel explanation (SHAP feature contributions for THIS specific "
            f"prediction): {json.dumps(forecast_result['explanation'])}\n"
            "Positive contribution values push the prediction higher, negative "
            "values push it lower."
        )
    elif forecast_result and forecast_result.get("model_used") == "transformer":
        explanation_text = (
            "\n\nNote: per-prediction feature explanations (SHAP) are only "
            "available for the XGBoost and LightGBM models in this system, not "
            "the Transformer. If asked why, say this plainly rather than guessing."
        )

    if GROQ_API_KEY:
        from groq import Groq
        client = Groq(api_key=GROQ_API_KEY)
        system_prompt = (
            "You are a malaria prevalence research assistant built on a real, "
            "evaluated forecasting system. You must ONLY use the numeric forecast "
            "data, SHAP explanation, and retrieved context given to you -- never "
            "invent numbers, dates, or claims not present in the provided data. "
            "If data is missing, say so plainly rather than guessing. Cite which "
            "source (project findings or PubMed) any non-numeric claim comes from. "
            "Respond in English. "
            f"{REGISTER_INSTRUCTIONS.get(register, REGISTER_INSTRUCTIONS['researcher'])} "
            "\n\nStructure every answer in exactly three labeled sections, on their own "
            "lines:\n"
            "Estimate: one or two sentences stating the numeric result and interval, if available.\n"
            "Reasoning: explain WHY, referencing the specific SHAP feature contributions "
            "by name and direction if given, and naming which retrieved finding "
            "supports any claim about climate, interventions, or model choice. Be "
            "concrete and specific, not generic.\n"
            "Caveats: one sentence on what this estimate does NOT account for "
            "(e.g. no live data feed, model-specific limitations, small region pool), "
            "only if genuinely relevant to this answer.\n"
            "Do not skip the Reasoning section even for short questions."
        )
        user_prompt = (
            f"User question: {parsed['raw_query']}\n\n"
            f"Forecast data (use these exact numbers if present):\n{forecast_text}"
            f"{explanation_text}\n\n"
            f"Retrieved context:\n{context_text}\n\n"
            "Answer the user's question using only the above."
        )
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            max_tokens=800,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return completion.choices[0].message.content

    # ---- deterministic fallback if no API key is configured ----
    parts = []
    if forecast_result:
        parts.append(
            f"For {forecast_result['admin']}, {forecast_result['country']}, the current "
            f"estimated prevalence is {forecast_result['current_estimate']:.1%}, with a "
            f"90% interval of {forecast_result['interval_90_lower']:.1%} to "
            f"{forecast_result['interval_90_upper']:.1%}, using the "
            f"{forecast_result['model_used']} model."
        )
    else:
        parts.append("No specific region matched your question.")
    parts.append(
        "(Set GROQ_API_KEY on the server to enable full LLM-composed, "
        "SHAP-grounded answers -- free at console.groq.com.)"
    )
    return " ".join(parts)


@app.post("/ask")
def ask(req: AskRequest):
    parsed = parse_query(req.query, model_store.regions_df)

    forecast_result = None
    if parsed["region_id"] is not None:
        density = _density_lookup()
        routing_decision = route(parsed["country"], density)
        try:
            forecast_result = predict_current_status(
                parsed["region_id"], routing_decision["model"], model_store
            )
            forecast_result["routing_reason"] = routing_decision["reason"]
        except ValueError:
            forecast_result = None

    retrieved = rag_index.retrieve(req.query, k=4)
    answer = _compose_answer(parsed, forecast_result, retrieved, req.register)

    seen, sources = set(), []
    for r in retrieved:
        if r["source"] not in seen:
            seen.add(r["source"])
            sources.append({"source": r["source"], "url": r["url"]})

    return {
        "answer": answer,
        "parsed_query": parsed,
        "forecast_data": forecast_result,
        "sources": sources,
        "llm_used": bool(GROQ_API_KEY),
    }


# ----------------------------------------------------------------------
# Static frontend
# ----------------------------------------------------------------------
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.isdir(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

    @app.get("/")
    def serve_index():
        return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))
