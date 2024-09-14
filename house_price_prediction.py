# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Sample dataset
data = {
    'GrLivArea': [1450, 1700, 1200, 1850, 2100, 1600, 1300, 2200, 1800, 2500],
    'Bedroom': [3, 4, 2, 3, 4, 3, 2, 5, 3, 4],
    'FullBath': [2, 2, 1, 2, 3, 2, 1, 3, 2, 3],
    'HalfBath': [1, 1, 0, 1, 1, 0, 0, 1, 1, 2],
    'SalePrice': [200000, 250000, 150000, 300000, 350000, 220000, 180000, 400000, 330000, 450000]
}

# Convert the data to a DataFrame
df = pd.DataFrame(data)

# Define the feature columns (independent variables) and target column (dependent variable)
features = ['GrLivArea', 'Bedroom', 'FullBath', 'HalfBath']
target = 'SalePrice'

# Split the data into features (X) and target (y)
X = df[features]
y = df[target]

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Output the evaluation results
print("Mean Squared Error:", mse)
print("R-squared:", r2)

# Optional: Display the coefficients of the model
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

# Predicting the price of a new house
new_house_data = pd.DataFrame([[2000, 3, 2, 1]], columns=features)  # New house features with column names
predicted_price = model.predict(new_house_data)
print("Predicted SalePrice for the new house:", predicted_price[0])

# Plotting the Actual vs. Predicted Sale Prices
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue', label='Predicted vs Actual')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='Perfect Prediction Line')
plt.xlabel('Actual SalePrice')
plt.ylabel('Predicted SalePrice')
plt.title('Actual vs. Predicted House SalesPrice')
plt.legend()
plt.grid(True)
plt.show()

