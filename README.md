# House Price Prediction using Linear Regression

## Overview

This project predicts house prices using a Linear Regression model. The model is trained on housing data and uses three important features:

* Area (Square Footage)
* Number of Bedrooms
* Number of Bathrooms

The goal is to estimate the price of a house based on these features.

## 🔗 Project Links

🚀 **Live Demo:**
[House Price Prediction App](https://housepriceprediction-uk4gscjfplpq8q7fuqugfk.streamlit.app/)

💻 **GitHub Repository:**
[House Price Prediction Repository](https://github.com/Chithra2936/HousePricePrediction)


## Technologies Used

* Python
* Pandas
* Scikit-learn
* Matplotlib
* Streamlit

## Dataset

The dataset contains housing information such as price, area, bedrooms, bathrooms, and other property-related features.

For this project, the following features were selected:

* Area
* Bedrooms
* Bathrooms

Target Variable:

* Price

## Project Workflow

1. Load the dataset using Pandas.
2. Select relevant features and target variable.
3. Split the dataset into training and testing sets.
4. Train a Linear Regression model.
5. Evaluate the model using regression metrics.
6. Predict house prices for new input values.
7. Visualize actual vs predicted house prices.

## Model Performance

* Mean Absolute Error (MAE): 1,265,275.67
* Mean Squared Error (MSE): 2,750,040,479,309.05
* R² Score: 0.4559

## Sample Prediction

Input:

* Area: 2000 sq ft
* Bedrooms: 3
* Bathrooms: 2

Predicted Price:

* 4,675,650.79

## How to Run

Install the required libraries:

```bash
pip install -r requirements.txt
```

Run the project:

```bash
python house_price_prediction.py
```

Run the Streamlit web app:

```bash
streamlit run app.py

Github Link:https://github.com/Chithra2936/HousePricePrediction

Live Demo: https://mall-customer-segmentation-aux2zx6tvywnpvrvdqtxrd.streamlit.app/

## Conclusion

This project demonstrates the implementation of Linear Regression for house price prediction. It provides a practical example of applying machine learning techniques to real-world housing data.

## Internship Task

SkillCraft Technology – Machine Learning Internship

Task 1: House Price Prediction using Linear Regression
