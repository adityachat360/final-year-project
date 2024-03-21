import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Load your dataset (replace 'thyroid_clean.csv' with the actual path)
data = pd.read_csv('thyroid_clean.csv')

# Assuming 'mal' is the column indicating the presence of thyroid cancer
X = data.drop(['mal', 'id'], axis=1)  # Features, 'id' is assumed to be an identifier column
y = data['mal']  # Target variable

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize a Support Vector Machine model
svm_clf = SVC(kernel='linear', random_state=42)

# Feature selection using SVM-based feature importance
sfm_svm = SelectFromModel(svm_clf, threshold='median')
sfm_svm.fit(X_train, y_train)

# Get the indices of the selected features based on importance
selected_feature_indices_svm = sfm_svm.get_support(indices=True)

# Use only the selected features
X_train_selected_svm = X_train.iloc[:, selected_feature_indices_svm]
X_test_selected_svm = X_test.iloc[:, selected_feature_indices_svm]

# Train the model on the selected features using k-fold cross-validation
svm_clf_selected = SVC(kernel='radial', random_state=42)
cv_scores = cross_val_score(svm_clf_selected, X_train_selected_svm, y_train, cv=5, scoring='accuracy')

# Print cross-validation scores
print("Cross-Validation Scores (SVM):", cv_scores)

# Fit the model on the full training set
svm_clf_selected.fit(X_train_selected_svm, y_train)

# Make predictions on the test set using the model with selected features
y_pred_selected_svm = svm_clf_selected.predict(X_test_selected_svm)

# Calculate precision, recall, F1-score, and support
precision, recall, f1_score, support = precision_recall_fscore_support(y_test, y_pred_selected_svm, average='binary')

# Print the evaluation metrics
print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1_score)
print("Support:", support)
