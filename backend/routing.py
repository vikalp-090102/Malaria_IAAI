"""
Routing logic, based on the density findings from evaluation:
- Countries with a large, well-surveyed region pool (roughly 15+ regions)
  route to the Transformer, since it needs a large pooled training set to
  be reliable (confirmed on Kenya, Tanzania, Nigeria).
- Smaller or sparser countries (e.g. Uganda, 8 regions) route to whichever
  of XGBoost/LightGBM scored better in per-country testing, since boosting
  models stayed stable everywhere tested, including small-country cases.
- Graph Neural ODE is never used to serve predictions; it is documented
  only as a negative result.
"""

DENSITY_THRESHOLD = 15

# from the pooled, per-country filtered results (fill in with your real numbers)
BEST_BOOSTING_MODEL_BY_COUNTRY = {
    "Kenya": "lightgbm",
    "Uganda": "lightgbm",
    "Tanzania": "xgboost",
    "Mozambique": "lightgbm",
    "Nigeria": "lightgbm",
    "Democratic Republic of the Congo": "xgboost",
}


def route(country: str, country_density: dict) -> dict:
    """
    country_density: {country_name: n_regions} lookup.
    Returns {"model": str, "reason": str}.
    """
    n_regions = country_density.get(country)

    if n_regions is None:
        return {
            "model": "lightgbm",
            "reason": (
                f"No region-density data found for '{country}'. Defaulting to "
                "LightGBM, the most broadly stable model across every tested "
                "configuration in this project."
            ),
        }

    if n_regions >= DENSITY_THRESHOLD:
        return {
            "model": "transformer",
            "reason": (
                f"{country} has {n_regions} surveyed regions in the training pool, "
                f"above the {DENSITY_THRESHOLD}-region threshold where the Transformer "
                "was found to be reliable. Routed to the pooled Transformer."
            ),
        }

    chosen = BEST_BOOSTING_MODEL_BY_COUNTRY.get(country, "lightgbm")
    return {
        "model": chosen,
        "reason": (
            f"{country} has only {n_regions} surveyed regions, below the "
            f"{DENSITY_THRESHOLD}-region threshold. The Transformer was found to be "
            f"unstable on small region pools in testing, so this request is routed "
            f"to {chosen}, which remained accurate across every small-country case "
            "tested, including Uganda."
        ),
    }
