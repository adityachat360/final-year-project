import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from imblearn.over_sampling import SMOTE

# Load your dataset (replace 'thyroid_clean.csv' with the actual path)
data = pd.read_csv('thyroid_clean.csv')

# Assuming 'mal' is the column indicating the presence of thyroid cancer
X = data.drop(['mal', 'id'], axis=1)  # Features, 'id' is assumed to be an identifier column
y = data['mal']  # Target variable

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply SMOTE to the training set
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Feature selection using Gradient Boosting Machine
gbm_clf = GradientBoostingClassifier(random_state=42)
sfm_gbm = SelectFromModel(gbm_clf, threshold='median')
X_train_selected_gbm = sfm_gbm.fit_transform(X_train_resampled, y_train_resampled)
X_test_selected_gbm = sfm_gbm.transform(X_test)

# Train the GBM model on the selected features using 10-fold cross-validation
cv_scores_gbm = cross_val_score(gbm_clf, X_train_selected_gbm, y_train_resampled, cv=10, scoring='accuracy')

# Print cross-validation scores
print("Cross-Validation Scores (GBM with SMOTE):", cv_scores_gbm)
print("Mean Cross-Validation Accuracy (GBM with SMOTE):", cv_scores_gbm.mean())

# Fit the GBM model on the full training set
gbm_clf.fit(X_train_selected_gbm, y_train_resampled)

# Make predictions on the test set using the model with selected features
y_pred_selected_gbm = gbm_clf.predict(X_test_selected_gbm)

# Calculate accuracy, precision, recall, F1-score, and support
accuracy_gbm = accuracy_score(y_test, y_pred_selected_gbm)
precision_gbm, recall_gbm, f1_score_gbm, support_gbm = precision_recall_fscore_support(y_test, y_pred_selected_gbm, average='binary')

# Print the evaluation metrics
print("Accuracy (GBM with SMOTE):", accuracy_gbm)
print("Precision (GBM with SMOTE):", precision_gbm)
print("Recall (GBM with SMOTE):", recall_gbm)
print("F1-Score (GBM with SMOTE):", f1_score_gbm)
print("Support (GBM with SMOTE):", support_gbm)
