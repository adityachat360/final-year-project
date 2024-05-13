import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

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

# Create a base classifier (Logistic Regression)
base_clf = LogisticRegression()

# Initialize RFE
rfe = RFE(estimator=base_clf, n_features_to_select=10, step=1)

# Fit RFE
rfe.fit(X_train_resampled, y_train_resampled)

# Transform the data
X_train_selected_backward_elimination = rfe.transform(X_train_resampled)
X_test_selected_backward_elimination = rfe.transform(X_test)

# Train the SVM model on the selected features
svm_clf = SVC()
svm_clf.fit(X_train_selected_backward_elimination, y_train_resampled)

# Make predictions on the test set using the model with selected features
y_pred = svm_clf.predict(X_test_selected_backward_elimination)

# Calculate accuracy, precision, recall, f1-score, and support
accuracy = accuracy_score(y_test, y_pred)
precision, recall, f1_score, support = precision_recall_fscore_support(y_test, y_pred, average='binary')

# Print the evaluation metrics
print("Accuracy (SVM with Backward Elimination + SMOTE):", accuracy)
print("Precision (SVM with Backward Elimination + SMOTE):", precision)
print("Recall (SVM with Backward Elimination + SMOTE):", recall)
print("F1-Score (SVM with Backward Elimination + SMOTE):", f1_score)
print("Support (SVM with Backward Elimination + SMOTE):", support)
