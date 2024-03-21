import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
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

# Feature selection using Random Forest
rf_clf = RandomForestClassifier(random_state=42)
sfm_rf = SelectFromModel(rf_clf, threshold='median')
X_train_selected_rf = sfm_rf.fit_transform(X_train_resampled, y_train_resampled)
X_test_selected_rf = sfm_rf.transform(X_test)

# Train the Random Forest model on the selected features using 10-fold cross-validation
cv_scores_rf = cross_val_score(rf_clf, X_train_selected_rf, y_train_resampled, cv=10, scoring='accuracy')

# Print cross-validation scores
print("Cross-Validation Scores (Random Forest with SMOTE):", cv_scores_rf)
print("Mean Cross-Validation Accuracy (Random Forest with SMOTE):", cv_scores_rf.mean())

# Fit the Random Forest model on the full training set
rf_clf.fit(X_train_selected_rf, y_train_resampled)

# Make predictions on the test set using the model with selected features
y_pred_selected_rf = rf_clf.predict(X_test_selected_rf)

# Calculate accuracy, precision, recall, F1-score, and support
accuracy_rf = accuracy_score(y_test, y_pred_selected_rf)
precision_rf, recall_rf, f1_score_rf, support_rf = precision_recall_fscore_support(y_test, y_pred_selected_rf, average='binary')

# Print the evaluation metrics
print("Accuracy (Random Forest with SMOTE):", accuracy_rf)
print("Precision (Random Forest with SMOTE):", precision_rf)
print("Recall (Random Forest with SMOTE):", recall_rf)
print("F1-Score (Random Forest with SMOTE):", f1_score_rf)
print("Support (Random Forest with SMOTE):", support_rf)
