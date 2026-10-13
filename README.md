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

## 4. Bayes Theorem Principles

## 5. Bayesian Regression Implementation Codes

## 6. Model Comparison Results

<img width="1142" height="361" alt="results" src="https://github.com/user-attachments/assets/c3cfb107-1b5b-49f7-a406-4d4e0b0fd702" />

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

- The interval bounds for the first five data points are remarkbly consistent across all three models. For instance, the first observation's lower ETI bounds are nearly identical, and upper bounds range similarly near 5.56 to 5.60. This demonstrates that applying Ridge or Lasso regularizing priors introduces negligible variation in the predictive uncertainity distributions compared to the unstandardized baseline.
- 

## 8. Coefficients Interpretation

## 9. Conclusion

## 10. Future Works

