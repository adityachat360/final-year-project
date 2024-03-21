import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Load your dataset (replace 'thyroid_clean.csv' with the actual path)
data = pd.read_csv('thyroid_clean.csv')

# Assuming 'mal' is the column indicating the presence of thyroid cancer
X = data.drop(['mal', 'id'], axis=1)  # Features, 'id' is assumed to be an identifier column
y = data['mal']  # Target variable

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature selection using Logistic Regression with L1 regularization
logreg_clf = LogisticRegression(penalty='l1', solver='liblinear', random_state=42)
sfm_logreg = SelectFromModel(logreg_clf, threshold='median')
X_train_selected_logreg = sfm_logreg.fit_transform(X_train, y_train)
X_test_selected_logreg = sfm_logreg.transform(X_test)

# Train the Logistic Regression model on the selected features using k-fold cross-validation
cv_scores_logreg = cross_val_score(logreg_clf, X_train_selected_logreg, y_train, cv=5, scoring='accuracy')

# Print cross-validation scores
print("Cross-Validation Scores (Logistic Regression):", cv_scores_logreg)
print("Mean Cross-Validation Accuracy (Logistic Regression):", cv_scores_logreg.mean())

# Fit the Logistic Regression model on the full training set
logreg_clf.fit(X_train_selected_logreg, y_train)

# Make predictions on the test set using the model with selected features
y_pred_selected_logreg = logreg_clf.predict(X_test_selected_logreg)

# Calculate accuracy, precision, recall, F1-score, and support
accuracy_logreg = accuracy_score(y_test, y_pred_selected_logreg)
precision_logreg, recall_logreg, f1_score_logreg, support_logreg = precision_recall_fscore_support(y_test, y_pred_selected_logreg, average='binary')

# Print the evaluation metrics
print("Accuracy (Logistic Regression):", accuracy_logreg)
print("Precision (Logistic Regression):", precision_logreg)
print("Recall (Logistic Regression):", recall_logreg)
print("F1-Score (Logistic Regression):", f1_score_logreg)
print("Support (Logistic Regression):", support_logreg)
