import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import AdaBoostClassifier
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

# Feature selection using AdaBoost
adaboost_clf = AdaBoostClassifier(random_state=42)
sfm_adaboost = SelectFromModel(adaboost_clf, threshold='median')
X_train_selected_adaboost = sfm_adaboost.fit_transform(X_train_resampled, y_train_resampled)
X_test_selected_adaboost = sfm_adaboost.transform(X_test)

# Train the AdaBoost model on the selected features using 10-fold cross-validation
cv_scores_adaboost = cross_val_score(adaboost_clf, X_train_selected_adaboost, y_train_resampled, cv=10, scoring='accuracy')

# Print cross-validation scores
print("Cross-Validation Scores (AdaBoost with SMOTE):", cv_scores_adaboost)
print("Mean Cross-Validation Accuracy (AdaBoost with SMOTE):", cv_scores_adaboost.mean())

# Fit the AdaBoost model on the full training set
adaboost_clf.fit(X_train_selected_adaboost, y_train_resampled)

# Make predictions on the test set using the model with selected features
y_pred_selected_adaboost = adaboost_clf.predict(X_test_selected_adaboost)

# Calculate accuracy, precision, recall, F1-score, and support
accuracy_adaboost = accuracy_score(y_test, y_pred_selected_adaboost)
precision_adaboost, recall_adaboost, f1_score_adaboost, support_adaboost = precision_recall_fscore_support(y_test, y_pred_selected_adaboost, average='binary')

# Print the evaluation metrics
print("Accuracy (AdaBoost with SMOTE):", accuracy_adaboost)
print("Precision (AdaBoost with SMOTE):", precision_adaboost)
print("Recall (AdaBoost with SMOTE):", recall_adaboost)
print("F1-Score (AdaBoost with SMOTE):", f1_score_adaboost)
print("Support (AdaBoost with SMOTE):", support_adaboost)
