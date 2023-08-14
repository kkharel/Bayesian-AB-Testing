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

class Two_Variant_Model:
  
  def __init__(self, priors: BetaPrior):
    self.priors = priors 
    

  def create_model(self, data: List[BinomialData]) -> pm.Model: 
    
    num_variants = len(data)
    
    trials = []
    successes = []

    for i in data:
      trials.append(i.trials)
      successes.append(i.successes)

    with pm.Model() as model: 
      
      p = pm.Beta("p", alpha = self.priors.alpha, beta = self.priors.beta, shape = num_variants)
      
      obs = pm.Binomial("y", n = trials, p = p, shape = num_variants, observed = successes) 
      
      relative_uplift = pm.Deterministic("relative_uplift_B", p[1] / p[0] - 1)  
      
    return model


weak_prior = Two_Variant_Model(BetaPrior(alpha = 500, beta = 500))

strong_prior = Two_Variant_Model(BetaPrior(alpha = 7000, beta = 7000))

# simulated data with trials = 1 and successes  = 1, check how the model behaves
with weak_prior.create_model(data=[BinomialData(1, 1), BinomialData(1, 1)]):
  weak_prior_predictive = pm.sample_prior_predictive(samples=100000, return_inferencedata=False)

with strong_prior.create_model(data=[BinomialData(1, 1), BinomialData(1, 1)]):
  strong_prior_predictive = pm.sample_prior_predictive(samples=100000, return_inferencedata=False)


az.style.use("arviz-doc")
fig, axs = plt.subplots(2, 1, figsize=(8.5, 11), sharex=True)  
az.plot_posterior(weak_prior_predictive["relative_uplift_B"], ax=axs[0], textsize = 7, color = "green", hdi_prob = 0.95)
axs[0].set_title(f"Variant B vs. Variant A Relative Uplift Prior Predictive, {weak_prior.priors}", fontsize=7)
axs[0].set_xlabel("Relative Uplift", fontsize = 7)
axs[0].set_ylabel("Density", fontsize = 7)
axs[0].tick_params(axis="both", labelsize=7)  
axs[0].axvline(x=0, color="orange", linewidth=0.7)
az.plot_posterior(strong_prior_predictive["relative_uplift_B"], ax=axs[1], textsize = 7, color = "green", hdi_prob = 0.95)
axs[1].set_title(f"Variant B vs. Variant A Relative Uplift Prior Predictive, {strong_prior.priors}", fontsize=7)
axs[1].set_xlabel("Relative Uplift", fontsize = 7)
axs[1].set_ylabel("Density", fontsize = 7)
axs[1].tick_params(axis="both", labelsize=7)  
axs[1].axvline(x=0, color="orange", linewidth=0.7)


plt.savefig("Prior_Predictive_Check.jpg", format="jpg", dpi=300, bbox_inches="tight")

plt.show()

# Generatng data

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


def Two_Variant(
  variants: List[str],
  success_probability: List[float],
  samples_per_variant: int,
  weak_prior: BetaPrior,
  strong_prior: BetaPrior,
) -> None:
  
  generated = generate_binomial_data(variants, success_probability, samples_per_variant)
  
  data = []
  for i in variants:
    generated_data = generated[i]
    data.append(BinomialData(**generated_data.to_dict()))

  
  print("Generated Data:")
  print(generated)

  with Two_Variant_Model(priors=weak_prior).create_model(data):
    trace_weak = pm.sample(draws=11000, tune=1000, cores = 1, chains=4)  # discard first 1000 samples as burn in so that we draw samples from stationary distribution
    print("\nTrace for Weak Prior:")
    print(pm.summary(trace_weak))
    
  with Two_Variant_Model(priors=strong_prior).create_model(data):
    trace_strong = pm.sample(draws=11000, tune=1000, cores = 1, chains=4)  
    print("\nTrace for Strong Prior:")
    print(pm.summary(trace_strong))
    
  true_relative_uplift = success_probability[1] / success_probability[0] - 1
  print("\nTrue Relative Uplift:")
  print(true_relative_uplift)
  
  fig, axs = plt.subplots(2, 1, figsize=(7, 7), sharex=True)
  az.plot_posterior(trace_weak.posterior["relative_uplift_B"], ax=axs[0], textsize = 7, color = "green", hdi_prob = 0.95)
  axs[0].set_title(f"True Relative Uplift = {true_relative_uplift:.1%}, {weak_prior}", fontsize=7)
  axs[0].axvline(x=0, color="orange")
  az.plot_posterior(trace_strong.posterior["relative_uplift_B"], ax=axs[1], textsize = 7, color = "green", hdi_prob = 0.95)
  axs[1].set_title(f"True Relative Uplift = {true_relative_uplift:.1%}, {strong_prior}", fontsize=7)
  axs[1].axvline(x=0, color="orange")
  fig.suptitle("Variant B vs. Variant A Relative Uplift", fontsize = 7)
  plt.savefig("Different_Success_Rate.jpg", format="jpg", dpi=300, bbox_inches="tight")
  plt.show()  
  return trace_weak, trace_strong


# Same Success Rate
trace_weak, trace_strong = Two_Variant(
  variants = ["A", "B"],
  success_probability = [0.20,0.20],
  samples_per_variant = 100000,
  weak_prior = BetaPrior(alpha = 500, beta = 500),
  strong_prior = BetaPrior(alpha = 7000, beta = 7000),
)


# Different Success Rates
trace_weak, trace_strong = Two_Variant(
  variants = ["A", "B"],
  success_probability = [0.25, 0.27],
  samples_per_variant = 100000,
  weak_prior = BetaPrior(alpha = 500, beta = 500),
  strong_prior = BetaPrior(alpha = 7000, beta = 7000),
)

