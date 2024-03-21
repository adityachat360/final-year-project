import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score

# Load your dataset (replace 'thyroid_clean.csv' with the actual path)
data = pd.read_csv('thyroid_clean.csv')

# Assuming 'mal' is the column indicating the presence of thyroid cancer
X = data.drop(['mal', 'id'], axis=1)  # Features, 'id' is assumed to be an identifier column
y = data['mal']  # Target variable

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize a Gradient Boosting Classifier
gbm_clf = GradientBoostingClassifier(n_estimators=100, random_state=42)

# Train the Gradient Boosting model
gbm_clf.fit(X_train, y_train)

# Feature selection using Gradient Boosting with the mean as the threshold
sfm = SelectFromModel(gbm_clf, threshold='mean')
sfm.fit(X_train, y_train)

# Get the indices of the top 10 selected features based on importance
selected_feature_indices = sfm.get_support(indices=True)[:10]

# Print selected feature indices
print("Selected Feature Indices:", selected_feature_indices)

# Get the names of the selected features
selected_feature_names = X.columns[selected_feature_indices]
print("Selected Feature Names:", list(selected_feature_names))

# Use only the top 10 selected features
X_train_selected = X_train.iloc[:, selected_feature_indices]
X_test_selected = X_test.iloc[:, selected_feature_indices]

# Train the model on the selected features
gbm_clf_selected = GradientBoostingClassifier(n_estimators=100, random_state=42)
gbm_clf_selected.fit(X_train_selected, y_train)

# Make predictions on the test set using the model with selected features
y_pred_selected = gbm_clf_selected.predict(X_test_selected)

# Evaluate the model with selected features
accuracy_selected = accuracy_score(y_test, y_pred_selected)
print("Accuracy with Top 10 Selected Features:", accuracy_selected)
