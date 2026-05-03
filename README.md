# WTI Crude Oil vs S&P 500 Market Analysis

This document outlines the methodology and functionality of a Python-based data analysis script designed to explore the historical relationship between West Texas Intermediate (WTI) Crude Oil prices and the S&P 500 index. By analyzing the last 10 years of monthly returns, the script evaluates how oil price fluctuations correlate with the broader stock market.

## Overview of the Analysis

The script is structured into four distinct analytical phases, moving from raw data processing to predictive modeling:

### 1. Data Processing & Preparation
The analysis begins by ingesting historical market data from an Excel file. The script performs the following transformations:
* **Date Standardization:** Converts raw observation dates into a standardized datetime format.
* **Time-Series Windowing:** Extracts a trailing 10-year window (120 months) for contemporary analysis.
* **Return Calculation:** Computes the monthly percentage returns for both WTI Crude and the S&P 500 to ensure the data is stationary and suitable for statistical modeling.

### 2. Outlier Mitigation
To prevent extreme market anomalies from skewing the underlying trends, the script implements a robust cleaning process:
* Calculates Z-scores for the monthly returns of both assets.
* Filters out extreme "black swan" events by removing data points that fall outside a 3-standard-deviation threshold.

### 3. Exploratory Data Analysis (EDA)
The script visually explores the relationship between the two assets through:
* **Scatter Plot Analysis:** Maps WTI Returns against S&P 500 Returns, complete with a calculated line of best fit to visualize the linear trend.
* **Correlation Heatmap:** Generates a Pearson correlation heatmap to quantify the strength and direction of the linear relationship between the two variables.

### 4. Statistical Modeling
To draw concrete statistical insights, the script applies two distinct modeling techniques:
* **Linear Regression (OLS):** Constructs an Ordinary Least Squares regression to predict the continuous S&P 500 return based on the WTI Crude return, yielding metrics such as R-squared, coefficients, and p-values.
* **Logistic Regression:** Transforms the S&P 500 returns into a binary outcome (positive/flat months vs. negative months) and trains a logistic regression model. This evaluates whether oil price movements can reliably classify the directional movement of the broader market.

## Generated Outputs

Upon execution, the script produces a comprehensive suite of analytical metrics and visualizations:
* **Data Cleaning Summary:** A detailed count of original data points versus the cleaned dataset.
* **Visualizations:** Renderings of the scatter plot and the correlation heatmap.
* **OLS Regression Summary:** A detailed statsmodels table breaking down the linear relationship.
* **Classification Metrics:** A confusion matrix and a classification report (Precision, Recall, F1-Score) detailing the logistic regression model's predictive accuracy.