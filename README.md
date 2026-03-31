# Bank Customer Churn Prediction

Predicting whether a bank customer will churn (leave) or stay using an Artificial Neural Network built with Keras and TensorFlow.

## Dataset
Churn Modelling Dataset from Kaggle  
Download: https://www.kaggle.com/datasets/shubh0799/churn-modelling  
Place `Churn_Modelling.csv` in the project root before running.

## Features Used
- Credit Score
- Geography (France / Germany / Spain)
- Gender
- Age
- Tenure
- Balance
- Number of Products
- Has Credit Card
- Is Active Member
- Estimated Salary

## Model Architecture
- Input layer: 10 features
- Hidden layer 1: 64 neurons, ReLU
- Hidden layer 2: 32 neurons, ReLU
- Output layer: 1 neuron, Sigmoid (binary classification)

## Training
- Optimizer: Adam
- Loss: Binary Crossentropy
- Epochs: 100
- Validation split: 20%

## Metric
- Accuracy

## Result
Model achieves ~86% accuracy on test data
