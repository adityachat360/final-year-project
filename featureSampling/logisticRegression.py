import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from imblearn.over_sampling import SMOTE
import numpy as np

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

# Compute Pearson correlation between each feature and the target variable
pearson_correlation = np.abs([np.corrcoef(X_train_resampled.iloc[:, i], y_train_resampled)[0, 1] for i in range(X_train_resampled.shape[1])])

# Select top k features based on highest absolute correlation
k = 10  # Select top 10 features (adjust k as needed)
selected_features = pearson_correlation.argsort()[-k:][::-1]
X_train_selected = X_train_resampled.iloc[:, selected_features]
X_test_selected = X_test.iloc[:, selected_features]

# Train the Logistic Regression model on the selected features using 10-fold cross-validation
logreg_clf = LogisticRegression(penalty='l1', solver='liblinear', random_state=42)
cv_scores_logreg = cross_val_score(logreg_clf, X_train_selected, y_train_resampled, cv=10, scoring='accuracy')

# Print cross-validation scores
print("Cross-Validation Scores (Logistic Regression with SMOTE and Pearson Correlation):", cv_scores_logreg)
print("Mean Cross-Validation Accuracy (Logistic Regression with SMOTE and Pearson Correlation):", cv_scores_logreg.mean())

# Fit the Logistic Regression model on the full training set
logreg_clf.fit(X_train_selected, y_train_resampled)

# Make predictions on the test set using the model with selected features
y_pred_selected_logreg = logreg_clf.predict(X_test_selected)

# Calculate accuracy, precision, recall, F1-score, and support
accuracy_logreg = accuracy_score(y_test, y_pred_selected_logreg)
precision_logreg, recall_logreg, f1_score_logreg, support_logreg = precision_recall_fscore_support(y_test, y_pred_selected_logreg, average='binary')

# Print the evaluation metrics
print("Accuracy (Logistic Regression with SMOTE and Pearson Correlation):", accuracy_logreg)
print("Precision (Logistic Regression with SMOTE and Pearson Correlation):", precision_logreg)
print("Recall (Logistic Regression with SMOTE and Pearson Correlation):", recall_logreg)
print("F1-Score (Logistic Regression with SMOTE and Pearson Correlation):", f1_score_logreg)
print("Support (Logistic Regression with SMOTE and Pearson Correlation):", support_logreg)
