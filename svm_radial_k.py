import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Load your dataset (replace 'thyroid_clean.csv' with the actual path)
data = pd.read_csv('thyroid_clean.csv')

# Assuming 'mal' is the column indicating the presence of thyroid cancer
X = data.drop(['mal', 'id'], axis=1)  # Features, 'id' is assumed to be an identifier column
y = data['mal']  # Target variable

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature selection using RFE with SVM
svm_clf = SVC(kernel='rbf', random_state=42)
n_features = min(5, X.shape[1])  # Select up to 5 features or all available features if less than 5
rfe = RFE(estimator=svm_clf, n_features_to_select=n_features, step=1)
X_train_selected_svm = rfe.fit_transform(X_train, y_train)
X_test_selected_svm = rfe.transform(X_test)

# Train the SVM model on the selected features using 10-fold cross-validation
cv_scores_svm = cross_val_score(svm_clf, X_train_selected_svm, y_train, cv=10, scoring='accuracy')

# Print cross-validation scores
print("Cross-Validation Scores (SVM with RBF kernel):", cv_scores_svm)
print("Mean Cross-Validation Accuracy (SVM with RBF kernel):", cv_scores_svm.mean())

# Fit the SVM model on the full training set
svm_clf.fit(X_train_selected_svm, y_train)

# Make predictions on the test set using the model with selected features
y_pred_selected_svm = svm_clf.predict(X_test_selected_svm)

# Calculate accuracy, precision, recall, F1-score, and support
accuracy_svm = accuracy_score(y_test, y_pred_selected_svm)
precision_svm, recall_svm, f1_score_svm, support_svm = precision_recall_fscore_support(y_test, y_pred_selected_svm, average='binary')

# Print the evaluation metrics
print("Accuracy (SVM with RBF kernel):", accuracy_svm)
print("Precision (SVM with RBF kernel):", precision_svm)
print("Recall (SVM with RBF kernel):", recall_svm)
print("F1-Score (SVM with RBF kernel):", f1_score_svm)
print("Support (SVM with RBF kernel):", support_svm)
