import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ==========================
# Load Dataset
# ==========================
data = pd.read_csv("train.csv")

print("Dataset Loaded Successfully!")
print("\nFirst 5 Rows:")
print(data.head())

# ==========================
# Dataset Information
# ==========================
print("\nDataset Shape:")
print(data.shape)

print("\nDataset Info:")
print(data.info())

print("\nMissing Values:")
print(data.isnull().sum())

# ==========================
# Feature Selection
# ==========================
X = data[['area', 'bedrooms', 'bathrooms']]
y = data['price']

# ==========================
# Train-Test Split
# ==========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)

# ==========================
# Model Training
# ==========================
model = LinearRegression()
model.fit(X_train, y_train)

print("\nModel Training Completed!")

# ==========================
# Predictions
# ==========================
y_pred = model.predict(X_test)

# ==========================
# Model Evaluation
# ==========================
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation")
print("-" * 40)
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R2 Score: {r2:.4f}")

# ==========================
# Model Coefficients
# ==========================
print("\nModel Coefficients")
print("-" * 40)

for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature}: {coef:.2f}")

print(f"Intercept: {model.intercept_:.2f}")

# ==========================
# Predict New House Price
# ==========================
new_house = pd.DataFrame({
    'area': [2000],
    'bedrooms': [3],
    'bathrooms': [2]
})

predicted_price = model.predict(new_house)

print("\nHouse Price Prediction")
print("-" * 40)
print("Area: 2000 sq ft")
print("Bedrooms: 3")
print("Bathrooms: 2")
print(f"Predicted Price: {predicted_price[0]:.2f}")

# ==========================
# Actual vs Predicted Plot
# ==========================
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    'r--'
)

plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices")

plt.tight_layout()

# Save Graph
plt.savefig("house_price_prediction.png")

plt.show()

print("\nGraph saved as: house_price_prediction.png")
print("Project Completed Successfully!")