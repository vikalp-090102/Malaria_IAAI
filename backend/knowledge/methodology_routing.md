# Forecasting methodology and model routing

Forecasts are produced with a walk-forward, one-year-ahead protocol. To forecast a
given year, each model is retrained using only real survey data from before that
year, then evaluated on that year alone. This mirrors how a health ministry would
actually use these forecasts in practice: retraining each year as new survey data
arrives, rather than forecasting many years into the future from a single fixed
snapshot. Early experiments that forecast five to seven years ahead from one fixed
training cutoff produced misleadingly flat, unrealistic-looking forecasts, because
the models had no way to anchor to a changing local situation.

Raw individual surveys are aggregated into a region-year panel, weighted by the
number of people examined in each survey, since raw point-level malaria prevalence
surveys are highly irregular in both timing and location, and averaging within a
region and year substantially reduces sampling noise.

This system routes each forecast request to the model most likely to be reliable
for that specific region, based on findings from extensive testing:

- If the requested region belongs to a country with a large, well-surveyed pool of
  regions (this system used a threshold of roughly fifteen to twenty regions and
  confirmed the effect down to individual countries such as Kenya, Tanzania, and
  Nigeria), the request is served by the Transformer, since it needs a large pooled
  training set to be reliable and benefits from climate covariates where available.
- If the region belongs to a smaller or sparser country (for example Uganda, which
  has only eight surveyed regions in this dataset), or if the region has little to
  no prior survey history at all, the request is served by XGBoost or LightGBM,
  since these models remained stable and accurate across every tested scenario,
  including small-country and unseen-region generalization tests.
- The Graph Neural ODE is never used to serve live forecasts, since it was
  consistently the least reliable model across every configuration tested.

Every forecast is returned with a calibrated ninety percent prediction interval, not
a bare point estimate. Raw uncertainty estimates from bootstrap ensembles and Monte
Carlo dropout substantially underestimated real forecast error in testing, likely
because malaria surveys themselves contain irreducible sampling noise that model-
level uncertainty methods do not capture. Intervals shown by this system have been
rescaled against historical forecast errors so that roughly ninety percent of past
forecasts would have fallen inside the stated range.
