# ISYE-6420 - Bayesian Statistics - Spring 2024
##  Final Project - My Own Bayesian Data Analysis
## Predicting Metacritic scores of Steam games with Bayesian models using DistilBERT for feature engineering

Saad M.Siddiqui (ssiddiqui60@gatech.edu)

## Overview 
A comparative analysis of Bayesian and conventional frequentist statistical learning methods for predicting Metacritic video game scores with DistilBERT and PCA-based custom feature engineering for unstructured data.

This was completed as a capstone project as part of a course in Bayesian Statistics for a graduate degree in Computer Science from Georgia Institute of Technology.

## Quick Links
- [Report](./isye_6420_project_report_ssiddiqui60.pdf)
- [Code](./isye_6420_project_code_ssiddiqui60.ipynb)

## Dataset Used
Subset of [FronkonGames' Steam Games Dataset](https://huggingface.co/datasets/FronkonGames/steam-games-dataset). 

First accessed on 2024-04-10.

## Caveats
- JAX-powered NumPyro NUTS sampling led to poor model fits across all models.
- Some models can take more several hours to fit without NumPyro. 
- You may find it useful to fit models on multiple concurrent Colab sessions.
- Posterior predictive sampling on outsample data led to errors for some models.
    - M2 (Poisson): exploding lambdas 
    - M3 (Negative Binomial): p < n 
    - Workaround was to implement posterior predictive sampling manually with NumPy using mean as Bayes estimator of regression coefficients and retracing transformations in the PyMC models.
