import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load your dataset (replace 'thyroid_clean.csv' with the actual path)
data = pd.read_csv('thyroid_clean.csv')

# Assuming 'mal' is the column indicating the presence of thyroid cancer
X = data.drop(['mal', 'id'], axis=1)  # Features, 'id' is assumed to be an identifier column
y = data['mal']  # Target variable

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Linear Regression model using 10-fold cross-validation
linreg = LinearRegression()
cv_scores_linreg = cross_val_score(linreg, X_train, y_train, cv=10, scoring='neg_mean_squared_error')

# Print cross-validation scores
print("Cross-Validation Scores (Linear Regression):", cv_scores_linreg)
print("Mean Cross-Validation MSE (Linear Regression):", -cv_scores_linreg.mean())

# Fit the Linear Regression model on the full training set
linreg.fit(X_train, y_train)

# Make predictions on the test set
y_pred_linreg = linreg.predict(X_test)

# Calculate evaluation metrics
mse_linreg = mean_squared_error(y_test, y_pred_linreg)
mae_linreg = mean_absolute_error(y_test, y_pred_linreg)
r2_linreg = r2_score(y_test, y_pred_linreg)

# Print the evaluation metrics
print("Mean Squared Error (Linear Regression):", mse_linreg)
print("Mean Absolute Error (Linear Regression):", mae_linreg)
print("R^2 Score (Linear Regression):", r2_linreg)
