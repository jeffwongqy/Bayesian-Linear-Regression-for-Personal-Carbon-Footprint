# Bayesian Regression for Personal Carbon Footprint

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

## 4. Bayes' Theorem 
At the heart of the method is Bayes' Theorem, which updates our initial assumptions using observed data: 



$$P(\theta | D) = \frac{P(D | \theta) P(\theta)}{P(D)}$$


where:
- $P(\theta)$ (Prior): What we believe about the parameters $\theta$ before looking at the data
- $P(D | \theta)$ (Likelihood): The probability of seeing our actual data given specific parameter values.
- $P(\theta | D)$ (Posterior): Our updated belief about the parameters after combining our prior knowledge with the observed data.
- $P(D)$ (Evidence): A normaliing constant representing the total probability of the data.

### 4.1 Setting Up the Linear Model
Mathematically, a Bayesian regression model connects inputs (X) to an output (y) using probability distribution. We assume that the target variable $y_{i}$ follows a Normal (Gaussian) distribution around a mean $\mu_{i}$: 

$$y_{i} \sim N(\mu_i, \sigma^{2})$$

The mean $\mu_{i}$ is calculated using the standard linear combination of features, intercept $(\beta_{0})$ and coefficients $(\beta_{j})$:

$$\mu_{i} = \beta_{0} + \sum^{k}_{j=1} X_{ij}\beta_{j}$$

### 4.2 Adding Priors for Regularization 
To prevent overfitting and handle noisy data, we assign prior distributions to the coefficients $(\beta)$:

#### 4.2.1 Normal Priors (L2 Regularization):

$$\beta_{j} \sim N(0, \sigma^{2}_{\beta})$$

This shrinks large coefficients smoothly toward zero, assuming most effects are small. 

### 4.2.2 Laplce Priors (L1 Regularization):

$$\beta_{j} \sim Laplce(0, b)$$

This pushes uninformative coefficients strictly to zero, effectively performing feature selection. 

## 5. Bayesian Regression Implementation Codes

## 6. Model Comparison Results

<img width="800" height="250" alt="results" src="https://github.com/user-attachments/assets/c3cfb107-1b5b-49f7-a406-4d4e0b0fd702" />

- All three models demonstrate strong generalization performance on the test set, explanining over 92% of the variance in personal carbon footprint.
- The standard unregularized Bayesian regression model slgihtly outperforms the regularized Ridge and Lasso variants across all metrics on this particular test split (achieving a lower MAE, lower RMSE, and higher R2). However, the performance differences between the three models are minimal, indicating that the feature space does not suffer from extreme multicollinearity requiring heavy shrinkage. 

## 7. Posterior Predictive Checks

### 7.1 PPC Plot
<img width="1397" height="481" alt="PPC" src="https://github.com/user-attachments/assets/fea52f36-5f67-465c-8b7d-6ce69cfe3963" />

- Across all three models (Standard, Ridge, and Lasso bayesian Regression), the posterior predictive distributions (blue lines) and their mean (orange dashed curve) track the observed data distribution (solid black curve) remarkbly well. This indicates that the models have successfully captured the underlying data-generating process for personal carbon footprints.
- The peak density - centered around a caron footprint value of 7.5 to 8.0 is accurately captured by the posterior repliaces in all three variants. The dispersion, spread, and overall bell-shaped contour closely mirror the empirical distribution.
- There are visible graphical discrepancies or structural differences among the Standard, Ridge, and Lasso PPC plots.

### 7.2 PPC HDIs and ETIs

<img width="830" height="751" alt="PPC2" src="https://github.com/user-attachments/assets/155cdec8-9c6c-4e7b-8f0b-030f4c03e1b0" />

- The interval bounds for the first five data points are remarkbly consistent across all three models. For instance, the first observation's lower ETI bounds are nearly identical, and upper bounds range are similar as well. This demonstrates that applying Ridge or Lasso regularizing priors introduces negligible variation in the predictive uncertainity distributions compared to the unstandardized baseline.
- The interval widths reflect the inherent residual variance captured by the models, providing robust upper and lower bounds for individual personal carbon footprint predictions at a 95% confidence level.


## 8. Coefficients Interpretation

<img width="1357" height="756" alt="model_coefficients" src="https://github.com/user-attachments/assets/ddb721f1-88b6-44ea-b755-903966ccb507" />

- The posterior means, standard deviations, and 94% highest density intervals for all coefficients and intercepts across the Standard, Ridge, and Lasso bayesian regression models are virtually identical, indicating that the choice of regularizing prior had a negligible impact on coefficient shrinkage for this dataset.


## 9. Conclusion
- All three models achieved exceptional generalisation capability on the test set, explaining over 92% of the variance with low error metrics such as MAE and RMSE.
- Posterior diagnostics confirmed reliable sampling behaviour, with all parameters achieving optimal convergence and high effective sample sizes.
- The introduction of Ridge and Lasso regularizing priors produced nearly identical coefficient estimates and predictive intervals compared to the unregularized Standard Bayesian model, indicating a well-conditioned feature space free of severe multicollinearity.


## 11. Datasets
https://www.kaggle.com/datasets/sonalshinde123/personal-carbon-footprint-behavior-dataset 

