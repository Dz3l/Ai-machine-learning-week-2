import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import urllib.request
import socket

# Set timeout for socket operations
socket.setdefaulttimeout(10)

# Download dataset with error handling
url = "https://raw.githubusercontent.com/owid/co2-data/master/owid-co2-data.csv"
try:
    data = pd.read_csv(url)
except urllib.error.URLError as e:
    print(f"Error downloading data: {e}. Please check your internet connection or try again later.")
    exit(1)

# Filter relevant columns and handle missing data
data = data[['year', 'country', 'co2', 'gdp', 'population', 'energy_per_capita']].dropna()

# Select features and target
features = ['gdp', 'population', 'energy_per_capita']
target = 'co2'

# Split data into training and testing sets
X = data[features]
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Absolute Error: {mae:.2f} million tonnes")
print(f"R2 Score: {r2:.2f}")

# Visualize results
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("Actual CO2 Emissions (million tonnes)")
plt.ylabel("Predicted CO2 Emissions (million tonnes)")
plt.title("Actual vs Predicted CO2 Emissions")
plt.savefig("co2_prediction_plot.png")
plt.show()

# Save model coefficients
coefficients = pd.DataFrame(model.coef_, features, columns=['Coefficient'])
print("\nModel Coefficients:")
print(coefficients)