import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# ==========================
# Load Dataset
# ==========================
data = pd.read_csv("train.csv")

# ==========================
# Feature Selection
# ==========================
X = data[['area', 'bedrooms', 'bathrooms']]
y = data['price']

# ==========================
# Train Model
# ==========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

# ==========================
# Streamlit UI
# ==========================
st.set_page_config(page_title="House Price Prediction", page_icon="🏠")

st.title("🏠 House Price Prediction")
st.write(
    "Predict house prices based on Area, Number of Bedrooms, and Number of Bathrooms using Linear Regression."
)

# User Inputs
area = st.number_input(
    "Enter Area (sq ft)",
    min_value=500,
    max_value=20000,
    value=2000
)

bedrooms = st.number_input(
    "Enter Number of Bedrooms",
    min_value=1,
    max_value=10,
    value=3
)

bathrooms = st.number_input(
    "Enter Number of Bathrooms",
    min_value=1,
    max_value=10,
    value=2
)

# Prediction Button
if st.button("Predict House Price"):
    input_data = pd.DataFrame({
        'area': [area],
        'bedrooms': [bedrooms],
        'bathrooms': [bathrooms]
    })

    prediction = model.predict(input_data)

    st.success(
        f"Predicted House Price: ₹ {prediction[0]:,.2f}"
    )

# Dataset Preview
with st.expander("View Dataset Sample"):
    st.dataframe(data.head())

st.markdown("---")
st.markdown("Developed by **Chithra Kurma**")