# Bayesian Linear Regression for Personal Carbon Footprint

<img width="768" height="354" alt="footprint-graphic-shutterstock_1608534475-scaled-e1643965744134-768x354" src="https://github.com/user-attachments/assets/8759115c-a6a6-47c1-8313-f32b66b16043" />

## 1. Introduction
Bayesian regression is a probabilistic approach to linear regression where statistical inference is placed within the context of Bayesian statistics. Instead of computing single point estimates (such as Ordinary Least Squares), parameters (coefficients and error variance) are treated as random variables characterized by probability distributions. By incorporating prior beliefs and updating them using observed data via Bayes' Theorem, Bayesian regression produces posterior distributions that quantify parameter uncertainty, naturally guard against overfitting through regularization priors, and facilitate robust predictions. 

## 2. Scopes
* Environmental and sustainability analytics, specifically modeling personal carbon footprint metrics based on behavioural, lifestyle, and resource usage indicators.
* Comparative analysis of three Bayesian regression formulations: Original Bayesian Regression, Bayesian Ridge Regression, and bayesian Lasso Regression.
* Assessment via model performance (R2, MSE, RMSE, MAE), posterior predictive checks, and coefficient parameter interpretation.

## 3. Objectives
- Formulate and fit Bayesian linear regression models using Markov Chain Monte Carlo (MCMC) sampling.
- Examine the mathematical forundations governing Bayesian inference and prior regularization.
- Compare model predictive accuracy and error metrics across the model variants.
- Interpret posterior distributions, credible intervals, and regression coefficients to derive actionable insights without relying on preprocessing steps.

## 4. Bayesian Linear Regression: Mathematical Formulas and Python Code

The most important formula connecting your entire Python implementation to Bayes' Theorem is:

$$P(\theta \mid D) = \frac{P(D \mid \theta) P(\theta)}{P(D)}$$

where:
- Prior $P(\theta)$: Initial assumptions about the regression parameters.
- Likelihood $P(D \mid \theta)$: Probability of observing the actual carbon footprint data given the parameters.
- Evidence $P(D)$: Overall probability of observing the data.
- Posterior $P(\theta \mid D)$: Updated probability distribution of the regression parameters after considering the observed data.

| Bayesian Concept | Mathematical Formula | Python Code | Brief Meaning |
| :--- | :--- | :--- | :--- |
| **Prior: Intercept** | $\beta_{o} \sim N(0, 10²)$ | `pm.Normal("intercept", mu=0, sigma=10)` | Sets the initial assumption about the regression intercept before observing the data. |
| **Prior: Regression coefficients** | $\beta_{j} \sim N(0, 10^{2})$ | `pm.Normal("beta", mu=0, sigma=10, shape=X_train.shape[1])` | Sets initial assumptions about the coefficients for each input variable, such as electricity and transportation. |
| **Prior: Standard deviation** | $\sigma \sim HalfNormal(1)$ | `pm.HalfNormal("sigma", sigma=1)` | Defines the prior distribution of the regression error's standard deviation. It must be positive. |
| **Linear regression equation** | $\mu_{i} = \beta_{0} + \sum_{j=1}^{p}x_{ij}\beta_{j}$ | `mu = intercept + pm.math.dot(X_train, beta)` | Calculates the predicted carbon footprint using the input variables and regression coefficients. |
| **Likelihood function** | $y_{i} \sim N(\mu_{i}, \sigma^{2})$ | `pm.Normal("carbon_footprint", mu=mu, sigma=sigma, observed=y_train)` | Measures how likely the observed carbon footprint data are, given the predicted values and model parameters. |
| **Posterior distribution (Bayes' Theorem)** | $P (\theta \mid D) = \frac{P(D \mid \theta) P(\theta)}{P(D)}$ | `trace = pm.sample(2000, tune=1000, cores=2, random_state=42, return_inferencedata=True)` | Updates the prior beliefs using the observed data to estimate the posterior distributions of the model parameters. |
| **Posterior predictive distribution** | $P(y_{new} \mid D) = \int P(y_{new} \mid \theta) P(\theta \mid D) d\theta$ | `pm.sample_posterior_predictive(trace, extend_inferencedata=True)` | Generates predicted carbon footprint values using posterior samples while accounting for uncertainty. |

All in all, the PyMC model defines the prior distributions and likelihood. Bayes' Theorem determines the posterior distribution, while pm.sample() uses MCMC to approximate that posterior distribution.


## 5. Model Comparison Results

<img width="800" height="250" alt="results" src="https://github.com/user-attachments/assets/c3cfb107-1b5b-49f7-a406-4d4e0b0fd702" />

- All three models demonstrate strong generalization performance on the test set, explanining over 92% of the variance in personal carbon footprint.
- The standard unregularized Bayesian regression model slgihtly outperforms the regularized Ridge and Lasso variants across all metrics on this particular test split (achieving a lower MAE, lower RMSE, and higher R2). However, the performance differences between the three models are minimal, indicating that the feature space does not suffer from extreme multicollinearity requiring heavy shrinkage. 

## 6. Posterior Predictive Checks

### 6.1 PPC Plot
<img width="1397" height="481" alt="PPC" src="https://github.com/user-attachments/assets/fea52f36-5f67-465c-8b7d-6ce69cfe3963" />

- Across all three models (Standard, Ridge, and Lasso bayesian Regression), the posterior predictive distributions (blue lines) and their mean (orange dashed curve) track the observed data distribution (solid black curve) remarkbly well. This indicates that the models have successfully captured the underlying data-generating process for personal carbon footprints.
- The peak density - centered around a carbon footprint value of 7.5 to 8.0 is accurately captured by the posterior repliaces in all three variants. The dispersion, spread, and overall bell-shaped contour closely mirror the empirical distribution.
- There are visible graphical discrepancies or structural differences among the Standard, Ridge, and Lasso PPC plots.

### 6.2 PPC HDIs and ETIs

<img width="830" height="751" alt="PPC2" src="https://github.com/user-attachments/assets/155cdec8-9c6c-4e7b-8f0b-030f4c03e1b0" />

- The interval bounds for the first five data points are remarkbly consistent across all three models. For instance, the first observation's lower ETI bounds are nearly identical, and upper bounds range are similar as well. This demonstrates that applying Ridge or Lasso regularizing priors introduces negligible variation in the predictive uncertainity distributions compared to the unstandardized baseline.
- The interval widths reflect the inherent residual variance captured by the models, providing robust upper and lower bounds for individual personal carbon footprint predictions at a 95% confidence level.


## 7. Coefficients Interpretation

<img width="1357" height="756" alt="model_coefficients" src="https://github.com/user-attachments/assets/ddb721f1-88b6-44ea-b755-903966ccb507" />

- The posterior means, standard deviations, and 94% highest density intervals for all coefficients and intercepts across the Standard, Ridge, and Lasso bayesian regression models are virtually identical, indicating that the choice of regularizing prior had a negligible impact on coefficient shrinkage for this dataset.


## 8. Conclusion
- All three models achieved exceptional generalisation capability on the test set, explaining over 92% of the variance with low error metrics such as MAE and RMSE.
- Posterior diagnostics confirmed reliable sampling behaviour, with all parameters achieving optimal convergence and high effective sample sizes.
- The introduction of Ridge and Lasso regularizing priors produced nearly identical coefficient estimates and predictive intervals compared to the unregularized Standard Bayesian model, indicating a well-conditioned feature space free of severe multicollinearity.


## 11. Datasets
https://www.kaggle.com/datasets/sonalshinde123/personal-carbon-footprint-behavior-dataset 

