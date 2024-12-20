# TWSVM Implementation with Breast Cancer Dataset

This project implements a Twin Support Vector Machine (TWSVM) classifier for the Breast Cancer dataset using Python. It includes data preprocessing, granular ball generation using clustering, hidden matrix generation with Random Vector Functional Link (RVFL), and TWSVM training and evaluation.

---

## Table of Contents
1. [Introduction](#introduction)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Project Workflow](#project-workflow)
5. [How to Run](#how-to-run)
6. [Outputs](#outputs)

---

## Introduction

The **Twin Support Vector Machine (TWSVM)** is a machine learning algorithm that constructs two non-parallel hyperplanes for binary classification. This project demonstrates TWSVM's effectiveness on the Breast Cancer dataset, incorporating:
- **Granular Ball Generation:** Using KMeans to cluster data.
- **Hidden Matrix Transformation:** Enhanced feature representation with RVFL.
- **Model Evaluation:** Classification accuracy as the key performance metric.

---

## Prerequisites

- **Python Version:** 3.6+
- **Required Libraries:**
  - `numpy`
  - `pandas`
  - `scikit-learn`
  - `cvxopt`

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/jriyyya/CV-project 
   cd CV-project
   cd CancerDatabase
   ```

2. Install dependencies
   ```bash
   pip install numpy pandas scikit-learn cvxopt
   ```

## Project Workflow

### 1. Load and Preprocess Data
- **Function:** `load_and_preprocess_data()`
- Description:
  - Loads the Breast Cancer dataset using `scikit-learn`.
  - Splits the dataset into training and testing sets (80% train, 20% test).

---

### 2. Granular Ball Generation
- **Function:** `generate_granular_balls(X, y, purity_threshold=0.95)`
- Description:
  - Applies KMeans clustering to form granular balls.
  - Retains clusters with purity above the specified threshold.

---

### 3. Hidden Matrix Transformation
- **Function:** `generate_hidden_matrix(X, N=23, activation_function=1)`
- Description:
  - Transforms the input data using RVFL with random weights and biases.
  - Supports the following activation functions:
    - **SELU:** Scaled Exponential Linear Unit
    - **ReLU:** Rectified Linear Unit
    - **Sigmoid**

---

### 4. TWSVM Training and Prediction
- **Function:** `TWSVM_main(DataTrain, TestX, d1, d2)`
- Description:
  - Solves two quadratic programming problems to identify hyperplanes for classification.
  - Predicts labels on the test dataset.

---

### 5. Evaluate Model
- **Description:**
  - Combines predictions with ground truth labels to compute classification accuracy.

## How to Run

#### 1. Ensure the dataset and required libraries are set up.

#### 2. Run the script
``` bash
python3 main.py
```


## Outputs

![image](https://github.com/user-attachments/assets/5523d9dc-f725-446f-8dc9-b6c2b53073c0)

