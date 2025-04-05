import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
file_path = r"C:\Users\vaish\Downloads\amazon.csv\meesho Orders Aug.csv"  # Replace with your actual file path
df = pd.read_csv(r"C:\Users\vaish\Downloads\amazon.csv\meesho Orders Aug.csv")

# Ensure column names are properly formatted
df.columns = df.columns.str.strip()

# Select features and target column
features = ["Quantity", "Supplier Discounted Price (Incl GST and Commision)"]
target = "Supplier Listed Price (Incl. GST + Commission)"

# Drop missing values
df = df.dropna(subset=features + [target])

# Split dataset into train and test sets
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Model Coefficients:", model.coef_)
print("Model Intercept:", model.intercept_)
print("Mean Squared Error:", mse)
print("R² Score:", r2)
