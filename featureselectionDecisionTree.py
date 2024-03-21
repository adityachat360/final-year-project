import pandas as pd
import numpy as np  # Add this import for NumPy
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score

# Load your dataset (replace 'thyroid_clean.csv' with the actual path)
data = pd.read_csv('thyroid_clean.csv')

# Assuming 'mal' is the column indicating the presence of thyroid cancer
X = data.drop(['mal', 'id'], axis=1)  # Features, 'id' is assumed to be an identifier column
y = data['mal']  # Target variable

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize a Decision Tree Classifier as the base estimator
base_dt_clf = DecisionTreeClassifier(random_state=42)

# Initialize a Bagging Classifier with Decision Tree as the base estimator
bagging_dt_clf = BaggingClassifier(base_dt_clf, n_estimators=10, random_state=42)
bagging_dt_clf.fit(X_train, y_train)

# Extract feature importances from individual trees
feature_importances = []
for tree in bagging_dt_clf.estimators_:
    feature_importances.append(tree.feature_importances_)

# Calculate mean feature importance across all trees
mean_feature_importance = np.mean(feature_importances, axis=0)

# Select features based on mean importance
sfm = SelectFromModel(base_dt_clf, threshold='mean')
sfm.fit(X_train, y_train)

# Transform the datasets
X_train_selected = sfm.transform(X_train)
X_test_selected = sfm.transform(X_test)

# Print selected feature indices
selected_feature_indices = sfm.get_support(indices=True)
print("Selected Feature Indices:", selected_feature_indices)

# Get the names of the selected features
selected_feature_names = X.columns[selected_feature_indices]
print("Selected Feature Names:", list(selected_feature_names))

# Train the model on the selected features
bagging_dt_clf_selected = BaggingClassifier(base_dt_clf, n_estimators=10, random_state=42)
bagging_dt_clf_selected.fit(X_train_selected, y_train)

# Make predictions on the test set using the model with selected features
y_pred_selected = bagging_dt_clf_selected.predict(X_test_selected)

# Evaluate the model with selected features
accuracy_selected = accuracy_score(y_test, y_pred_selected)
print("Accuracy with Selected Features (Bagging):", accuracy_selected)
