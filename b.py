import dask.dataframe as dd
import dask.array as da
from dask_ml.model_selection import train_test_split
from dask_ml.linear_model import LinearRegression
from dask_ml.preprocessing import StandardScaler

# Load large dataset from CSV
file_path = r"C:\Users\vaish\Downloads\amazon.csv\meesho Orders Aug.csv"  # Replace with your actual file path
df = dd.read_csv(r"C:\Users\vaish\Downloads\amazon.csv\meesho Orders Aug.csv")

# Ensure column names are properly formatted
df.columns = df.columns.str.strip()

# Select features and target column for regression
features = ["Quantity", "Supplier Discounted Price (Incl GST and Commision)"]
target = "Supplier Listed Price (Incl. GST + Commission)"

# Drop missing values
df = df.dropna(subset=features + [target])

# Convert to Dask arrays for ML processing
X = df[features].to_dask_array(lengths=True)
y = df[target].to_dask_array(lengths=True)

# Split dataset into train and test sets
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

# Print model coefficients and intercept
print("Model Coefficients:", model.coef_)
print("Model Intercept:", model.intercept_)
