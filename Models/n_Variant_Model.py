from dataclasses import dataclass
from typing import Dict, List, Union
from warnings import simplefilter

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm

from scipy.stats import bernoulli, expon

simplefilter("ignore")

rng = np.random.default_rng(1000) 


@dataclass 

class BetaPrior: 
  alpha: float
  beta: float

@dataclass

class BinomialData: 
  trials: int # represents number of attempts
  successes: int # represents number of successes out of some number of attempts


class n_variant_Model:
  def __init__(self, priors: BetaPrior):
      self.priors = priors
    
  def create_model(self, data: List[BinomialData], comparison_method) -> pm.Model:
      num_variants = len(data)
      trials = [i.trials for i in data]
      successes = [i.successes for i in data]

      with pm.Model() as model:
          p = pm.Beta("p", alpha=self.priors.alpha, beta=self.priors.beta, shape=num_variants)
          obs = pm.Binomial("y", n=trials, p=p, shape=num_variants, observed=successes)

          reluplift = []
          for i in range(num_variants):
              if comparison_method == "compare_to_control":
                  comparison = p[0]
              elif comparison_method == "best_among_all":
                  others = [p[j] for j in range(num_variants) if i != j]
                  if len(others) > 0:
                      comparison = others[0]
                      for j in range(1, len(others)):
                          comparison = pm.math.maximum(comparison, others[j])
                  else:
                      comparison = p[0]
              else:
                  raise ValueError(f"Comparison method {comparison_method} is not valid.")
              reluplift.append(pm.Deterministic(f"reluplift_{i}", p[i] / comparison - 1))
      return model

def generate_binomial_data(
  variants: List[str], success_probability: List[str], samples_per_variant: int 
) -> pd.DataFrame:
  data = {}
  for variant, p in zip(variants, success_probability):
    data[variant] = bernoulli.rvs(p, size = samples_per_variant)
  agg = (
    pd.DataFrame(data).aggregate(["count","sum"]).rename(index={"count":"trials", "sum": "successes"})
  )
  return agg #, pd.DataFrame(data)


def n_variant(
  variants: List[str],
  success_probability: List[float],
  samples_per_variant: int,
  weak_prior: BetaPrior,
  strong_prior: BetaPrior,
  comparison_method: str,
):
  generated = generate_binomial_data(variants, success_probability, samples_per_variant)
  
  data = []
  for i in variants:
    generated_data = generated[i]
    data.append(BinomialData(**generated_data.to_dict()))

  print("Generated Data:")
  print(generated)

  with n_variant_Model(priors=weak_prior).create_model(data=data, comparison_method=comparison_method):
    weak_trace = pm.sample(draws=11000, tune=1000, cores=1, chains=4, init="auto") 
    print("\nTrace for Weak Prior:")
    print(pm.summary(weak_trace))
    
  with n_variant_Model(priors=strong_prior).create_model(data=data, comparison_method=comparison_method):
    strong_trace = pm.sample(draws=11000, tune=1000, cores=1, chains=4, init="auto")
    print("\nTrace for Strong Prior:")
    print(pm.summary(strong_trace))
    
  n_plots = len(variants)
  
  fig, axs = plt.subplots(nrows=n_plots, ncols=2, figsize=(7, 7), sharex=True)

  for i, variant in enumerate(variants):
    if i == 0 and comparison_method == "compare_to_control":
      axs[i, 0].set_yticks([])
      axs[i, 1].set_yticks([])
    else:
      az.plot_posterior(weak_trace.posterior[f"reluplift_{i}"], ax=axs[i, 0], textsize=7, color="green", hdi_prob=0.95)
      axs[i, 0].set_title(f"Weak Prior: Relative Uplift {variant}, Success rate = {success_probability[i]:.2%}", fontsize=7)
      axs[i, 0].axvline(x=0, color="orange")
      
      az.plot_posterior(strong_trace.posterior[f"reluplift_{i}"], ax=axs[i, 1], textsize=7, color="green", hdi_prob=0.95)
      axs[i, 1].set_title(f"Strong Prior: Relative Uplift {variant}, Success rate = {success_probability[i]:.2%}", fontsize=7)
      axs[i, 1].axvline(x=0, color="orange")

  fig.suptitle(f"Method {comparison_method}", fontsize=7)
  
  plt.savefig("Best_Among_All.jpg", format="jpg", dpi=300, bbox_inches="tight")

  plt.show()  

  return weak_trace, strong_trace

# Compare One at a Time to control
weak_trace, strong_trace = n_variant(
    variants=["A", "B", "C", "D"],
    success_probability=[0.21, 0.23, 0.228, 0.26],
    samples_per_variant=100000,
    weak_prior = BetaPrior(alpha = 500, beta = 500),
    strong_prior = BetaPrior(alpha = 7000, beta = 7000),
    comparison_method="compare_to_control",
)

# Find the best among all
weak_trace, strong_trace = n_variant(
    variants=["A", "B", "C", "D"],
    success_probability=[0.21, 0.23, 0.228, 0.26],
    samples_per_variant=100000,
    weak_prior = BetaPrior(alpha = 500, beta = 500),
    strong_prior = BetaPrior(alpha = 7000, beta = 7000),
    comparison_method="best_among_all",
)
