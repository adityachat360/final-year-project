import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load your dataset (replace 'your_dataset.csv' with the actual path)
data = pd.read_csv('thyroid_clean.csv')

# Assuming 'mal' is the column indicating the presence of thyroid cancer
X = data.drop(['mal', 'id'], axis=1)  # Features, 'id' is assumed to be an identifier column
y = data['mal']  # Target variable

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize a Random Forest Classifier
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the Random Forest model
rf_clf.fit(X_train, y_train)

# Make predictions on the test set using Random Forest
y_pred_rf = rf_clf.predict(X_test)

# Evaluate the Random Forest model
accuracy_rf = accuracy_score(y_test, y_pred_rf)
classification_report_rf = classification_report(y_test, y_pred_rf)

# Print the results for Random Forest
print("Random Forest Results:")
print(f"Accuracy: {accuracy_rf:.4f}")
print("Classification Report:\n", classification_report_rf)

# Initialize an AdaBoost Classifier with a base estimator (e.g., Decision Tree)
ada_clf = AdaBoostClassifier(base_estimator=RandomForestClassifier(n_estimators=50), n_estimators=100, random_state=42)

# Train the AdaBoost model
ada_clf.fit(X_train, y_train)

# Make predictions on the test set using AdaBoost
y_pred_ada = ada_clf.predict(X_test)

# Evaluate the AdaBoost model
accuracy_ada = accuracy_score(y_test, y_pred_ada)
classification_report_ada = classification_report(y_test, y_pred_ada)

# Print the results for AdaBoost
print("\nAdaBoost Results:")
print(f"Accuracy: {accuracy_ada:.4f}")
print("Classification Report:\n", classification_report_ada)