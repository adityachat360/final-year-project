import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.feature_selection import RFE
from imblearn.combine import SMOTETomek

# Load your dataset (replace 'thyroid_clean.csv' with the actual path)
data = pd.read_csv('thyroid_clean.csv')

# Assuming 'mal' is the column indicating the presence of thyroid cancer
X = data.drop(['mal', 'id'], axis=1)  # Features, 'id' is assumed to be an identifier column
y = data['mal']  # Target variable

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply SMOTE-Tomek to the training set
smt = SMOTETomek(random_state=42)
X_train_resampled, y_train_resampled = smt.fit_resample(X_train, y_train)

# Initialize the GBM classifier
gbm_clf = GradientBoostingClassifier(random_state=42)

# Perform backward feature elimination
# Start with all features and gradually eliminate the least important ones
selector = RFE(estimator=gbm_clf, n_features_to_select=10, step=1)  # Select top 10 features
selector = selector.fit(X_train_resampled, y_train_resampled)

# Get the selected features
selected_features = X.columns[selector.support_]

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.5],
    'max_depth': [3, 5, 7, 9],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'subsample': [0.8, 0.9, 1.0],
    'max_features': ['auto', 'sqrt', 'log2']
}
grid_search = GridSearchCV(gbm_clf, param_grid, cv=3, n_jobs=-1)
grid_search.fit(X_train_resampled[selected_features], y_train_resampled)

# Train the GBM classifier on the selected features with the best hyperparameters
best_gbm_clf = grid_search.best_estimator_
best_gbm_clf.fit(X_train_resampled[selected_features], y_train_resampled)

# Make predictions on the test set using the model with selected features
y_pred = best_gbm_clf.predict(X_test[selected_features])

# Calculate accuracy, precision, recall, f1-score, and support
accuracy = accuracy_score(y_test, y_pred)
precision, recall, f1_score, support = precision_recall_fscore_support(y_test, y_pred, average='binary')

# Print the evaluation metrics
print("Selected Features:", selected_features)
print("Best Hyperparameters:", grid_search.best_params_)
print("Accuracy (GBM with Backward Elimination + Hyperparameter Tuning + SMOTE-Tomek):", accuracy)
print("Precision (GBM with Backward Elimination + Hyperparameter Tuning + SMOTE-Tomek):", precision)
print("Recall (GBM with Backward Elimination + Hyperparameter Tuning + SMOTE-Tomek):", recall)
print("F1-Score (GBM with Backward Elimination + Hyperparameter Tuning + SMOTE-Tomek):", f1_score)
print("Support (GBM with Backward Elimination + Hyperparameter Tuning + SMOTE-Tomek):", support)
