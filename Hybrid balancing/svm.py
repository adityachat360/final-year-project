import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from imblearn.combine import SMOTETomek
from sklearn.feature_selection import SelectKBest, chi2

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

# Feature selection using SelectKBest with chi-square
selector = SelectKBest(chi2, k=10)
X_train_selected = selector.fit_transform(X_train_resampled, y_train_resampled)
X_test_selected = selector.transform(X_test)

# Train an SVM classifier on the selected features
svm_clf = SVC(random_state=42)
svm_clf.fit(X_train_selected, y_train_resampled)

# Make predictions on the test set using the model with selected features
y_pred = svm_clf.predict(X_test_selected)

# Calculate accuracy, precision, recall, f1-score, and support
accuracy = accuracy_score(y_test, y_pred)
precision, recall, f1_score, support = precision_recall_fscore_support(y_test, y_pred, average='binary')

# Print the evaluation metrics
print("Accuracy (SVM with Chi-Square + SMOTE-Tomek):", accuracy)
print("Precision (SVM with Chi-Square + SMOTE-Tomek):", precision)
print("Recall (SVM with Chi-Square + SMOTE-Tomek):", recall)
print("F1-Score (SVM with Chi-Square + SMOTE-Tomek):", f1_score)
print("Support (SVM with Chi-Square + SMOTE-Tomek):", support)
