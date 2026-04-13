# F1 Podium Predictor

## Overview

### Product Summary

A machine learning-driven application that predicts Formula 1 podium finishes for a given season and compares those predictions against actual results through an interactive visualization interface.

The system combines data processing, feature engineering, model training, and front-end visualization into a single pipeline designed for both analysis and demonstration.

### Goal

Build an end-to-end data science and visualization project that demonstrates:

machine learning applied to real-world sports data
feature engineering and model comparison
interactive data visualization
full-stack integration of ML and frontend systems

## Objectives & Success Metrics

### Primary Objectives

Train models to predict podium counts per driver
Compare multiple machine learning approaches
Provide an intuitive visualization for predictions vs actual results
Maintain a clean and interpretable data pipeline

### Success Metrics

Strong predictive performance relative to baseline
Clear separation between model outputs
Visually intuitive comparison interface
Consistent results across multiple seasons

## Target Users

### Primary User

Recruiters and interviewers evaluating technical projects

### Secondary Use Cases

Formula 1 fans interested in analytics
Students learning machine learning and data visualization

## Key Features

### Data Processing

Load and clean historical F1 race data
Handle missing values and inconsistencies
Normalize and structure data for modeling

### Feature Engineering

Driver performance trends
Team performance indicators
Historical podium frequency
Season-based aggregations

### Model Training

Train multiple regression models including:
K-Nearest Neighbors
Random Forest
Compare performance across models
Evaluate using appropriate metrics

### Prediction Engine

Generate predicted podium counts for each driver
Support predictions for specific seasons
Output structured results for visualization

### Interactive Visualization

Display predicted vs actual podium counts
Enable comparison across drivers
Provide filtering or selection by season

## Tech Stack

Frontend: JavaScript, D3.js
Backend/Data Processing: Python, Pandas
Machine Learning: Scikit-learn

## How It Works

Data Collection
Historical race data is gathered and cleaned
Feature Engineering
Extract meaningful features such as driver consistency, team performance, and historical podium rates
Model Training
Train multiple machine learning models and evaluate them using appropriate metrics
Prediction
Generate predicted podium counts for each driver
Visualization
Display predictions versus actual results in an interactive interface

## Installation and Setup

Clone the repository and navigate into the project directory
Install dependencies using the requirements file
Run the training pipeline from the source directory
Open the frontend locally or serve it using a simple local server

## Example Output

Predicted podium counts per driver
Actual podium counts
Side-by-side comparison chart
Interactive filtering by year

## Model Performance

KNN: Moderate accuracy
Random Forest: Best overall performance
Additional models evaluated using cross-validation

Exact metrics depend on dataset and tuning

## Future Improvements

Add real-time race predictions
Incorporate weather and qualifying data
Improve model accuracy with more advanced models
Deploy as a full web application
Add user input for custom predictions