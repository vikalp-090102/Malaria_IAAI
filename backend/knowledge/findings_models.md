# Model comparison findings

Four architectures were compared on walk-forward, one-year-ahead malaria prevalence
forecasting across sub-Saharan Africa: XGBoost, LightGBM, a Graph Neural ODE, and a
Transformer with continuous-time embeddings.

XGBoost and LightGBM performed best overall and were the most stable across every
tested configuration, including isolated single-country training. Pooled and
filtered by country, these models reached R-squared values as high as 0.70 to 0.76
in well-surveyed countries such as Kenya and Tanzania, versus a naive baseline of
predicting each region's own historical average.

The Transformer performed competitively when trained on a large pooled dataset
across multiple countries, but was unstable and sometimes performed worse than a
naive baseline when trained in isolation on a single country, even a densely
surveyed one such as Kenya. Pooled training fixed this instability, raising Kenya's
R-squared from negative values to roughly 0.33 to 0.44 depending on whether climate
covariates were included.

The Graph Neural ODE was the weakest model in every configuration tested, including
after adding a spatial-neighbor feature, walk-forward retraining, and pooled
training across countries. In several cases, pooling actively worsened its
performance on individual countries, suggesting its graph-mixing mechanism does not
generalize well across regions from different climates and malaria ecologies mixed
into a single learned adjacency structure. It is not used for live forecasting in
this system, and is retained only as a documented negative result.
