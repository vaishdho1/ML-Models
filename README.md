# Exploring Machine Learning Algorithms Across Multiple Datasets

## Overview

This repository contains various machine learning projects and implementations. Each file represents a different project or algorithm implemented from scratch without using high-level libraries like `scikit-learn`,`TensorFlow` or `Pytorch`. Below are the details of each script and the datasets they utilize.
It also contains a project report which shows various hyper parameters used for tuning the models and the set of hyperparameters which give the best performance.
## Evaluation 
Accuracy and F1-score are the metrics used for evaluating the performance.
## Files and Datasets

- **RandomForest_digits.py**: Implementation of a Random Forest classifier on the digits dataset from scratch.
  - **Dataset**: The digits dataset is a part of the `sklearn` library and contains 8x8 pixel images of handwritten digits (0-9).
  - **Description**: The goal is to classify digits based on the pixel values.

- **RandomForest_parkinsons.py**: Implementation of a Random Forest classifier on the Parkinson's disease dataset from scratch.
  - **Dataset**: The Parkinson's dataset contains biomedical voice measurements from people with and without Parkinson's disease.
  - **Description**: The objective is to distinguish between healthy individuals and those with Parkinson's disease based on voice measurements.

- **decision_tree_mixed.py**: Implementation of a Decision Tree classifier on a mixed dataset from scratch.
  - **Dataset**: This dataset includes various features from multiple sources (specific details can be filled in if available).
  - **Description**: A decision tree model is used to classify instances based on mixed-type features.

- **k_fold_project_loan.py**: Implementation of K-Fold Cross Validation on a loan prediction dataset using models implemented from scratch.
  - **Dataset**: The loan prediction dataset contains information about loan applicants and whether they were approved for a loan.
  - **Description**: The aim is to predict loan approval status using K-Fold Cross Validation for model validation.

- **k_fold_project_titanic.py**: Implementation of K-Fold Cross Validation on the Titanic dataset using models implemented from scratch.
  - **Dataset**: The Titanic dataset from Kaggle contains data on the passengers of the Titanic, including demographics and whether they survived.
  - **Description**: The goal is to predict survival of passengers using various features like age, sex, and ticket class, validated with K-Fold Cross Validation.

- **knn_project.py**: Implementation of a K-Nearest Neighbors classifier on a sample dataset from scratch.
  - **Dataset**: A custom or standard dataset (specific details can be filled in if available).
  - **Description**: Using KNN to classify instances based on their proximity to other labeled instances.

- **neural_network.py**: Implementation of a Neural Network for a classification problem from scratch.
  - **Dataset**: The dataset used can vary; typically includes labeled data suitable for training a neural network.
  - **Description**: A neural network model is implemented to perform classification tasks.

- **preprocessor.py**: Script for data preprocessing including cleaning and transformation.
  - **Functionality**: This script is designed to preprocess data by handling missing values, encoding categorical variables, and normalizing features.

## Usage

To run any of the scripts, use the following command:
```bash
python <script_name>.py
