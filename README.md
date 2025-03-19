# Comparing-Classifiers

## Overview
This project aims to compare the performance of four classification models—K-Nearest Neighbors (KNN), Logistic Regression, Decision Trees, and Support Vector Machines (SVM)—on a dataset related to the marketing of bank products. The dataset contains information from multiple marketing campaigns conducted by a Portuguese banking institution based on the UC Irvine Machine Learning Repository.

## Goals
- Apply various classification methods to a business problem.
- Compare the results of K-Nearest Neighbors, Logistic Regression, Decision Trees, and Support Vector Machines.

## Dataset
The dataset is sourced from UCI Machine Learning Repository including:
- Client attributes (age, job, etc.)
- Campaign-related attributes (contact type, etc.)
- External factors (consumer price index, etc.)

## Project Structure
This repository contains:
- Comparing Classifiers in Banking through Marketing Analysis.ipynb – Jupyter Notebook with data preprocessing, model training, evaluation, and visualizations.
- README.md – Overview of the project, dataset, methodology, findings, and next steps.

## Methodology
1. Data Preprocessing:
   - Label Encoding
   - Imputation.
   - Standardized numerical features.

2. Model Training & Evaluation:
   - KNN, Logistic Regression, Decision Trees, and SVM.
   - Accuracy and AUC (Area Under the Curve).
   - Compared classification reports and confusion matrices.

3. Visualizations:
   - Bar plots to compare model performance.

## Findings & Insights
- Logistic Regression had the highest AUC (0.912), making it the best at predicting customer subscriptions.
- SVM also performed well with an AUC of 0.906.
- Decision Trees had lower AUC but better for business applications.
- KNN had strong accuracy but lower AUC.

## Recommendations
- Logistic Regression for its high AUC.
- Consider SVMfor high accuracy.
- Decision Trees for interpretability is required.
- Enhance results with hyperparameter tuning and ensemble methods.

## Next Steps
- New feature combinations to improve model performance.
- Optimize classifiers using GridSearchCV.
- Implement the best-performing model in a production environment.
