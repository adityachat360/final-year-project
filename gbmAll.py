import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Load your dataset (replace 'thyroid_clean.csv' with the actual path)
data = pd.read_csv('thyroid_clean.csv')

# Assuming 'mal' is the column indicating the presence of thyroid cancer
X = data.drop(['mal', 'id'], axis=1)  # Features, 'id' is assumed to be an identifier column
y = data['mal']  # Target variable

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize a Gradient Boosting Classifier
gbm_clf = GradientBoostingClassifier(n_estimators=100, random_state=42)

# Feature selection using Gradient Boosting with the mean as the threshold
sfm_gbm = SelectFromModel(gbm_clf, threshold='mean')
X_train_selected_gbm = sfm_gbm.fit_transform(X_train, y_train)
X_test_selected_gbm = sfm_gbm.transform(X_test)

# Train the GBM model on the selected features using k-fold cross-validation
cv_scores_gbm = cross_val_score(gbm_clf, X_train_selected_gbm, y_train, cv=5, scoring='accuracy')

# Print cross-validation scores
print("Cross-Validation Scores (GBM):", cv_scores_gbm)
print("Mean Cross-Validation Accuracy (GBM):", cv_scores_gbm.mean())

# Fit the GBM model on the full training set
gbm_clf.fit(X_train_selected_gbm, y_train)

# Make predictions on the test set using the model with selected features
y_pred_selected_gbm = gbm_clf.predict(X_test_selected_gbm)

# Calculate accuracy, precision, recall, F1-score, and support
accuracy_gbm = accuracy_score(y_test, y_pred_selected_gbm)
precision_gbm, recall_gbm, f1_score_gbm, support_gbm = precision_recall_fscore_support(y_test, y_pred_selected_gbm, average='binary')

# Print the evaluation metrics
print("Accuracy (GBM):", accuracy_gbm)
print("Precision (GBM):", precision_gbm)
print("Recall (GBM):", recall_gbm)
print("F1-Score (GBM):", f1_score_gbm)
print("Support (GBM):", support_gbm)
