
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt
import urllib.request
import socket

# Set timeout for socket operations
socket.setdefaulttimeout(10)

# Load data
url = "https://raw.githubusercontent.com/owid/co2-data/master/owid-co2-data.csv"
try:
    data = pd.read_csv(url)
except urllib.error.URLError as e:
    print(f"Error downloading data: {e}. Please check your internet connection or try again later.")
    exit(1)

# Show shape before any cleaning
print("Original data shape:", data.shape)

# Handle missing numerical data with median imputation
data = data.fillna(data.select_dtypes(include=np.number).median())

# Drop any columns still containing NaNs (e.g., columns that were entirely NaN)
data = data.dropna(axis=1, how='any')

# Define features and target
features = ['gdp', 'population', 'energy_per_capita']
target = 'co2'

# Check if required columns are present
required_columns = features + [target]
missing = [col for col in required_columns if col not in data.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# Drop rows with missing values in the selected columns (just to be safe)
data = data.dropna(subset=required_columns)
print("Cleaned data shape:", data.shape)

# Feature scaling
scaler = StandardScaler()
X = data[features]
y = data[target]
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"\nMean Absolute Error: {mae:.2f} million tonnes")
print(f"R2 Score: {r2:.2f}")

# Cross-validation
scores = cross_val_score(model, X_train, y_train, cv=5)
print(f"Cross-validation R2 scores: {scores.mean():.2f}")
print(f"Cross-validation standard deviation: {scores.std():.2f}")

# Plot: Actual vs Predicted
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
line_points = np.linspace(y_test.min(), y_test.max(), 100)
plt.plot(line_points, line_points, 'r--', lw=2)
plt.xlabel("Actual CO2 Emissions (million tonnes)")
plt.ylabel("Predicted CO2 Emissions (million tonnes)")
plt.title("Actual vs Predicted CO2 Emissions")

# Save plot
try:
    plt.savefig("co2_prediction_plot_improved.png")
except Exception as e:
    print(f"Error saving plot: {e}")

plt.show()

# Show coefficients
coefficients = pd.DataFrame(model.coef_, features, columns=['Coefficient'])
print("\nModel Coefficients:\n", coefficients)
