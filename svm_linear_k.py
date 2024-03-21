import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
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

# Feature selection using Linear SVM
svm_clf = SVC(kernel='linear', random_state=42)
sfm_svm = SelectFromModel(svm_clf, threshold='median')
X_train_selected_svm = sfm_svm.fit_transform(X_train_resampled, y_train_resampled)
X_test_selected_svm = sfm_svm.transform(X_test)

# Train the SVM model on the selected features using 10-fold cross-validation
cv_scores_svm = cross_val_score(svm_clf, X_train_selected_svm, y_train_resampled, cv=10, scoring='accuracy')

# Print cross-validation scores
print("Cross-Validation Scores (Linear SVM with SMOTE):", cv_scores_svm)
print("Mean Cross-Validation Accuracy (Linear SVM with SMOTE):", cv_scores_svm.mean())

# Fit the SVM model on the full training set
svm_clf.fit(X_train_selected_svm, y_train_resampled)

# Make predictions on the test set using the model with selected features
y_pred_selected_svm = svm_clf.predict(X_test_selected_svm)

# Calculate accuracy, precision, recall, F1-score, and support
accuracy_svm = accuracy_score(y_test, y_pred_selected_svm)
precision_svm, recall_svm, f1_score_svm, support_svm = precision_recall_fscore_support(y_test, y_pred_selected_svm, average='binary')

# Print the evaluation metrics
print("Accuracy (Linear SVM with SMOTE):", accuracy_svm)
print("Precision (Linear SVM with SMOTE):", precision_svm)
print("Recall (Linear SVM with SMOTE):", recall_svm)
print("F1-Score (Linear SVM with SMOTE):", f1_score_svm)
print("Support (Linear SVM with SMOTE):", support_svm)
