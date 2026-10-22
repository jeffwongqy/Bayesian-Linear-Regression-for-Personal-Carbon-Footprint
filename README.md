# Bayesian Linear Regression for Personal Carbon Footprint

<img width="768" height="354" alt="footprint-graphic-shutterstock_1608534475-scaled-e1643965744134-768x354" src="https://github.com/user-attachments/assets/8759115c-a6a6-47c1-8313-f32b66b16043" />

## 1. Introduction
Bayesian Regression is a probabilistic approach to linear regression that applies the principles of Bayesian statistics to statistical modelling and inference. Unlike traditional methods such as Ordinary Least Squares (OLS), which estimate model parameters as fixed values, Bayesian Regression treats the regression coefficients (β) and error variance (σ²) as random variables described by probability distributions.

By incorporating prior knowledge or assumptions about the model parameters and updating them with observed data through Bayes' Theorem, Bayesian Regression produces posterior probability distributions that quantify the uncertainty associated with each parameter. This approach provides a natural form of regularisation through prior distributions, helping to control overfitting and improve model generalisation. Furthermore, the posterior distributions enable probabilistic predictions, allowing the model to estimate not only expected outcomes but also the uncertainty surrounding those predictions.

## 2. Objectives
- To formulate and implement Bayesian linear regression models using Markov Chain Monte Carlo (MCMC) sampling techniques.
- To understand the mathematical foundations of Bayesian inference, including prior distributions, likelihood functions, posterior distributions, and prior-based regularisation.
- To compare the predictive performance of Standard Bayesian Regression, Bayesian Ridge Regression, and Bayesian Lasso Regression using established evaluation metrics.
- To interpret posterior distributions, credible intervals, and regression coefficients to identify the relationships between lifestyle factors and personal carbon footprint levels.
- To investigate how different prior distributions influence parameter estimation, model regularisation, and predictive performance.

## 3. Bayesian Linear Regression: Mathematical Formulas and Python Code

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

In summary, the posterior distribution represents the balance between the prior and the likelihood. When the sample size is small, the prior generally has a stronger influence on the posterior. As the sample size increases, the likelihood contributes more information, and the posterior becomes increasingly influenced by the observed data. During Bayesian inference, pm.sample() uses MCMC sampling to generate many samples from the posterior distribution. These samples help estimate the regression coefficients, quantify uncertainty, and understand the possible values of the model parameters.

**Lasso** and **Ridge Regression** can also be connected to Bayesian Linear Regression. The key difference is that Lasso and Ridge use **regularisation** to control regression coefficients, while Bayesian Regression uses **prior probability distributions** to influence the coefficients.

In fact, Ridge and Lasso can be interpreted from a Bayesian perspective by **choosing different prior distributions for the regression coefficients**.

| Bayesian Concept | Mathematical Formula | Python Code | Brief Meaning |
| :--- | :--- | :--- | :--- |
| **Prior: Bayesian Ridge Regression Coefficients** | $\beta_{j} \sim N(0, \tau^{2})$ | `beta = pm.Normal("beta", mu=0, sigma=tau, shape=p)` | Uses a Normal prior to encourage coefficients towards zero. This corresponds to L2 regularisation under MAP estimation.|
| **Prior: Bayesian Lasso Regression Coefficients** | $\beta_{j} \sim Laplace(0, b)$ | `beta = pm.Laplace("beta", mu=0, b=b, shape=p)` | Uses a Laplace prior to encourage coefficients towards zero, corresponding to L1 regularisation under MAP estimation.|


## 4. Model Comparison Results

<img width="800" height="250" alt="results" src="https://github.com/user-attachments/assets/c3cfb107-1b5b-49f7-a406-4d4e0b0fd702" />

- All three models demonstrate very similar predictive performance, with only negligible differences across the evaluation metrics.
- Standard Bayesian Regression performs slightly best overall, achieving the lowest MAE (0.5930) and MSE (0.5805), the highest R² (0.9231), and a low RMSE (0.7619).
- Bayesian Ridge Regression has marginally higher errors and a slightly lower R² (0.9230), indicating comparable predictive accuracy.
- Bayesian Lasso Regression also performs similarly, with an RMSE of 0.7620 and R² of 0.9230.
- Overall, the standard Bayesian Regression achieves the best numerical results, but the differences are extremely small. All three models explain approximately 92.3% of the variance in the target variable, suggesting that Ridge and Lasso regularisation provide no meaningful improvement in predictive accuracy for this dataset.


## 5. Posterior Predictive Checks
**Posterior Predictive Checking (PPC)** is a method used in Bayesian statistics to evaluate whether a fitted model can generate predictions that resemble the actual observed data.

In simple terms, it checks whether the model can reproduce the patterns and distribution of the original dataset.

How does it work?
1. **Generate predictions:** Use **pm.sample_posterior_predictive()** to generate simulated data based on samples from the posterior distribution.
2. **Compare distributions:** Compare the simulated predictions (posterior predictive) against the actual observed data.
3. **Assess model fit:** Check whether the predicted distributions closely match the observed distribution.

Why is it necessary?
- **Evaluate model fit:** Determines whether the model captures the patterns in the observed data.
- **Identify potential problems:** Highlights differences between predicted and observed distributions, such as incorrect spread, skewness, or extreme values.
- **Build confidence:** Helps assess whether the model is reasonably suitable for describing and predicting the data.

### 5.1 PPC Plot
Compares simulated data generated from the Bayesian model against the observed data to assess how well the model reproduces the actual data distribution.

<img width="1397" height="481" alt="PPC" src="https://github.com/user-attachments/assets/fea52f36-5f67-465c-8b7d-6ce69cfe3963" />

- Standard Bayesian Regression: The posterior predictive curves closely follow the observed distribution, indicating a good fit to the carbon footprint data.
- Bayesian Ridge Regression: The predictive curves also closely match the observed data, showing that Ridge regularisation maintains similar predictive performance.
- Bayesian Lasso Regression: The results are similarly aligned with the observed distribution, suggesting that Lasso regularisation also provides a good fit.

Overall: All three models produce very similar posterior predictive distributions, indicating that their predictions are consistent with the observed data, with no substantial difference in model fit.

### 5.2 PPC HDIs and ETIs

<img width="830" height="751" alt="PPC2" src="https://github.com/user-attachments/assets/155cdec8-9c6c-4e7b-8f0b-030f4c03e1b0" />

- Standard Bayesian Regression: The 95% ETI lower and upper bounds are approximately 7.07–10.10 for the first observation. Ridge and Lasso produce very similar intervals, indicating comparable predictive uncertainty across all three models.

Overall comparison: All three models have closely aligned 95% ETI bounds, suggesting that their posterior predictive distributions are very similar. Neither Ridge nor Lasso provides a substantial difference in predictive uncertainty compared with Standard Bayesian Regression.


## 6. Coefficients Interpretation

<img width="1357" height="756" alt="model_coefficients" src="https://github.com/user-attachments/assets/ddb721f1-88b6-44ea-b755-903966ccb507" />

- The posterior means, standard deviations, and 94% highest density intervals for all coefficients and intercepts across the Standard, Ridge, and Lasso bayesian regression models are virtually identical, indicating that the choice of regularizing prior had a negligible impact on coefficient shrinkage for this dataset.


## 7. Conclusion
- All three models achieved exceptional generalisation capability on the test set, explaining over 92% of the variance with low error metrics such as MAE and RMSE.
- Posterior diagnostics confirmed reliable sampling behaviour, with all parameters achieving optimal convergence and high effective sample sizes.
- The introduction of Ridge and Lasso regularizing priors produced nearly identical coefficient estimates and predictive intervals compared to the unregularized Standard Bayesian model, indicating a well-conditioned feature space free of severe multicollinearity.


## 8. Datasets
https://www.kaggle.com/datasets/sonalshinde123/personal-carbon-footprint-behavior-dataset 

