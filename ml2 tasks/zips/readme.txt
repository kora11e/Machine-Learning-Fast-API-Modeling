# Insurance Charges Prediction API

A FastAPI-based API for predicting insurance charges using pre-trained machine learning models: Random Forest, Decision Tree, and XGBoost. The API validates incoming feature data, tracks request/response analytics, and includes response time headers.

## Problem description:

### Goal: 
Find the most optimal representation of insurance charges, represented in a column "charges".

### Dataset: 
This dataset contains medical insurance cost information for 1338 individuals. It includes demographic and health-related variables such as age, sex, BMI, number of children, smoking status, and residential region in the US. The target variable is charges, which represents the medical insurance cost billed to the individual.

### Models used: 
- Random Forest, 
- Extreme Gradient Boost,
- Deep Neural Network (Linear, BatchNorm, RELu, Dropout)

### Metrics:
- Mean Squared error for comparison of models
- Mean Absolute Error for measuring charge inaccuracy
- Mean Absolute Percentage Error for measuring percentage charge inaccuracy
- R2 for measuring model variance

# Customer Segmentation Prediction for New Markets

## Context

An automobile company plans to enter new markets with existing products (P1-P5). Based on market research, the new market behavior is similar to their existing market. In the existing market, customers are classified into 4 segments (A, B, C, D), and this segmented outreach strategy has been highly successful. The goal is to predict the correct segment for 2627 new potential customers.

### Goal:
Develop a machine learning model to predict the customer segment (A, B, C, or D).

# How to run?

1. create local environment, either through venv or anaconda
2. isntall packages from requirements.txt
3. Run files and API
 
[Repository link](https://github.com/kora11e/Machine-Learning-Fast-API-Modeling/tree/main)
