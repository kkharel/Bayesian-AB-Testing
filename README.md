# Bayesian A/B Testing for Proportions with two Variant and n-Variant Use Cases

# Use Case: Two Variant

We want to know whether the new landing page results in more purchase than current landing page.

A - existing landing page.
B - new page we want to test.
metric of interest - proportion of users purchasing at least one item --> a bernoulli conversion - User either purchases at least one item(success) or do not purchase an item(failure).

The question we are trying to answer: Is variant B better than variant A?

## Assumptions:

Assume visitors behavior on landing page is independent of other visitors.

Assume we do not know the true success probabilities of each variant but instead they follow beta distribution. 

Assume probabilities are independent (split traffic between each variant randomly,so one variant would not affect the other variant).

Assume observed data (number of users landing on page and puchasing at least one item) follows a Bernoulli distribution.


## Metrics:

We look at the relative uplift in success probability to assess whether B is better than A that is, the percent difference in true success probability.


## Background: 

Bayes Formula:

P(conversion|data) = P(conversion)*P(data|conversion)/P(data)

where,

Initial Guess (P(conversion)): Start with initial estimate of how often visitors buy something on the website. This is the educated guess based on historical data or assumptions.

Actual Behavior (Likelihood of Data given conversion Rate): Then we see how well our initial estimate matches what's actually happening. We check how often people are really making purchases on the site.

Overall Patterns (Overall Likelihood): We consider how frequently people generally buy things online, regardless of the website. This gives us an idea of what's common on the internet.

Updated Estimate (Posterior Conversion Rate): Finally, we combine all this information to improve the initial estimate. This new estimate takes into account starting guess, the actual behavior of visitors, and the broader patterns of online shopping.


## Steps for Bayesian A/B testing:

### Define Prior Distribution: 
Start with initial assumption about the conversion rates before any data is collected. The prior distribution can be chosen based on domain knowledge, historical data, or it can be non-informative to avoid bias.

### Collect Data: 
Collect data on the number of conversions and the total number of trials (visitors, clicks, etc.) for each variant.

### Update the Prior with Likelihood: 
Model the likelihood of observing the data given the assumed conversion rates by using a probability distribution, often the binomial distribution for each variant.

### Calculate Posterior Distribution: 
Using Bayes theorem, update the prior beliefs with the likelihood calculated from the observed data which results in posterior distribution, representing updated beliefs about the conversion rates for each variant.

### Sample from the Posterior: 
Sample from posterior distribution using MCMC to get range of possible conversion rates.

### Analyze and Make Decisions: 
With samples from the posterior distribution, compute various statistics such as mean, median, credible intervals, etc.

### Compare and Choose: 
Compare the posterior distributions of the conversion rates for variants A and B. We can determine which variant is likely to have a higher conversion rate based on the mean or median of the posterior distributions or by comparing credible intervals.

### Monitor and Iterate: 
Bayesian A/B testing allows us to continuously update our beliefs as more data becomes available which in turn refines the posterior distribution and the confidence in true conversion rate increases.

## Concepts:

### Highest Density Interval, what is it?:
Highest Density Interval (HDI), also known as the credible interval, is a way to summarize the uncertainty around a parameter estimate. It provides a range of values within which the true parameter value is likely to fall with a certain level of confidence. The HDI is an interval that contains a specified percentage of the posterior distribution of a parameter. For example, a 95% HDI contains the central 95% of the posterior distribution.

### Interpretation: 
If we calculate the HDI for a parameter, it means that we can be X% confident that the 
true parameter value lies within that interval.

### Comparison of Intervals: 
If we are comparing HDIs of parameters between two (A/B) groups, such as conversion rates for variant A and variant B, and their HDIs do not overlap, it suggests that there's a significant difference between the two groups. If the HDIs overlap, it indicates that the difference between the groups might not be statistically significant.

### Width of HDI: 
A narrower HDI indicates that we have more precise estimates of the parameter, while a wider HDI indicates greater uncertainty.

### Robustness: 
HDIs are robust to the shape of the posterior distribution, unlike point estimates (e.g., mean or median), which can be influenced by outliers.

### Reporting: 
"We are Z% confident that the true conversion rate of variant A/B lies between X% and Y%."

In summary, HDIs in Bayesian A/B testing provide a probabilistic range within which the true parameter value is likely to fall, offering a more comprehensive and informative perspective on uncertainty and inference.


## Choosing the Prior?

We are considering the following to choose the prior:
  
We assume the same Beta prior is set for each variant initially
  
### Weakly informative prior:
We set low values for alpha and beta meaning we do not know anything about the value of the parameter, nor our confidence around it. We are interested in comparing the relative lift of one variant over another. With a weakly informative Beta prior, the relative uplift distribution is very wide meaning variants could be very different from each other.
  
### Strong informative prior:
We set high values for alpha and beta. The relative uplift distribution is thin meaning our prior belief is that the variants are not very different from each other.
  

## Prior Predictive Checks:

Prior predictive checks are conducted using two different prior settings (weak and strong priors) to simulate the expected behavior of the model under different assumptions about the conversion rates. 

By generating simulated data based on these priors and then analyzing the simulated data, we can gain insights into how the model responds to different prior beliefs and understand the range of outcomes we might expect in the absence of actual observed data. This can help us make informed decisions about the suitability of chosen priors and model structure.

In summary, It's a way to explore the impact of different prior beliefs on our model's predictions before seeing any actual data.

![Prior_Predictive_Check](https://github.com/kkharel/Bayesian-A-B-Testing/assets/59852121/816e3cc4-af5f-4a3b-9b08-c242f7df2e90)

With 95% HDI for relative uplift for B over A is roughly [-9%, +9%] for weak prior, with strong prior it is roughly [-2.0% +2%]. This will be the starting point for relative uplift distribution and will affect how observed conversion translate to posterior distribution. How we choose the prior depends on the context of the application.A strong prior helps us guard against false discoveries, but may require more data to detect the best variants. A weak prior gives more weight to the observed data but could also lead to more false discovieres as a result of early stopping.

Now, we consider two scenarios, one where the true success probability for each variant is the same and the other where Variant B has higher success probability than variant A.

For each variant and its corresponding true sucess probability (p), the model generates random samples using the bernoulli.rvs function from the SciPy library. This function simulates Bernoulli trials based on the success probability, p, and the specified number of samples (samples_per_variant).

Below is the simulation of the case when both variants have the same success probabilities

![Same_Success_Rate](https://github.com/kkharel/Bayesian-A-B-Testing/assets/59852121/521d4841-56de-4328-9de3-0739b59b7cd0)


We can see that for both weak prior and strong prior the true uplift of 0% lies within 95% HDI which suggests the decision that we should not roll out variant B.

Now, below is the simulation of the case when variant B have the higher true success probability than variant A

![Different_Success_Rate](https://github.com/kkharel/Bayesian-A-B-Testing/assets/59852121/a311f4f4-5bf2-483d-8ea1-eba2a7b454ca)

In both weak prior and strong prior case, the posterior relative uplift distribution suggests B
has a higher conversion rate than A, as the 95% HDI is well above 0. The decision here is to roll out variant B to all users and this outcome is "true discovery".

In practice, we are also interested in how much better is variant B than A?

For the model with strong prior, the prior is effectively pulling the relative uplift distribution closer to zero, so our central estimate of relative uplift is conservative( understated). We would need much more data for our inference to get closer to true relative uplift of 8%

# Use Case: n-Variant

 WHen we have more than two variants that we want to test simultaneously. We want to determine if any of these variants outperforms the others, 
 
 To achieve this, we explore two different methods

## Method 1: Compare to Control (Variant A)

In this method, we select one variant as the "control" variant which is a reference point. We then compare the performance of the other variants against this control variant, one at a time. The idea is to assess whether any of the other variants show a significant improvement or decline in performance compared to the control.

### Advantages
This method is straightforward to implement and interpret.It provides a clear comparison between each variant and the control, helping us identify variants that show a clear performance difference.

### Drawbacks
If there are multiple variants that outperform the control, it doesn't explicitly tell us which of these variants is the best among them. We can't make definitive inferences about the relative performance of variants that beat the control without additional analysis.

## Method 2: Best among All

In this method, we take a different approach by comparing each variant to the maximum performance among the other variants. This method addresses the drawback of the first method by identifying both the best and the worst performers among all variants, providing a more comprehensive understanding of their relative performance.

### Advantages:
It effectively finds the variant that shows the highest uplift in performance compared to all other variants. It accounts for situations where multiple variants are outperforming the control and helps identify the variant that performs best relative to the others.

### Drawbacks:
The calculation of relative uplift can be more complex compared to the simple comparison to the control. Depending on the data and distribution, this method might not always provide a clear distinction between the best and the second-best performers.

In summary, both methods have their pros and cons. Method 1 is simpler and provides a direct comparison to the control, which can be useful for initial assessments. Method 2, on the other hand, offers a more comprehensive understanding of relative performance but might require more complex calculations. It's important to consider the trade-offs and make an informed decision based on the insights we want to derive from the analysis.

Below is the simulation of the case when we treat A as control and compare all other variants with A one at a time.

![Compare_to_Control](https://github.com/kkharel/Bayesian-A-B-Testing/assets/59852121/7b0e3532-8e19-411b-bca2-349da012c897)


We can see that all the variants are btter than variant A but how do we know that among all the variants which one to choose? Suppose for example, if we do not have variant D then how would we choose between variant B and C? Best among all method implemented in the code handles such situations as well.

From the results, it suggests to roll out variant D. If we look at weak prior and strong prior, we see that strong prior understated our central estimate. we need more data for strong prior to get closer to true relative uplift of 26%

Now, 
Below is the simulation of a case where we compare all the variants and pick the best among all of them.
![Best_Among_All](https://github.com/kkharel/Bayesian-A-B-Testing/assets/59852121/ad37f0b2-abdb-4573-94e3-80f7efcc1eae)


We can clearly see that variant D outperforms all other variants. All other variants performs worst than variant D. Hence, we should roll out variant D in this case.

This wraps up Bayesian A/B testing method for Proportions

Citations:

Chris Stucchio. Bayesian a/b testing at vwo. 2015. URL: https://vwo.com/downloads/VWO\_SmartStats\_technical\_whitepaper.pdf.

John Kruschke. Doing Bayesian data analysis: A tutorial with R, JAGS, and Stan. Academic Press, 2014.

Initial Code Authors:
Cuong Duong, percevalve from pyMC. The initial code provided through pyMC has been modified here to capture relevant use cases and images.
