# Climate and intervention findings

Published research on malaria across Africa, notably Bhatt et al. (2015, Nature),
attributes most of the decline in Plasmodium falciparum prevalence between 2000 and
2015 to the scale-up of insecticide-treated bed nets and other interventions, rather
than to climate change. This system's own experiments are broadly consistent with
that finding.

Adding NASA POWER climate covariates (monthly temperature, precipitation, and
humidity) had a different effect on each model type. It had almost no effect on
XGBoost and LightGBM's accuracy. It measurably improved the Transformer's accuracy,
by roughly 17 percent in the six-country comparison, most likely because the
Transformer's attention mechanism can weigh an entire climate trajectory over time,
while boosting models treat each climate value as an independent feature with no
sense of sequence. It slightly worsened the Graph Neural ODE's accuracy in most
tests.

Sustained multi-year declines in prevalence, particularly the steep drops visible in
several Kenyan and Tanzanian regions starting around 2006 to 2008, line up in timing
with known national ITN and antimalarial drug scale-up campaigns rather than with
any climate shift in the same period. This system's models were not given direct
access to region-level intervention coverage, only national-level insecticide-
treated net coverage from the WHO Global Health Observatory, so they cannot fully
separate the effect of interventions from other unmodeled factors, and any
explanation involving interventions should be treated as informed context rather
than a proven causal claim from the model itself.
