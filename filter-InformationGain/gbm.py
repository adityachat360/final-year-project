import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectKBest, mutual_info_classif

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

# Feature selection using SelectKBest with mutual information gain
selector = SelectKBest(mutual_info_classif, k=10)
X_train_selected = selector.fit_transform(X_train_resampled, y_train_resampled)
X_test_selected = selector.transform(X_test)

# Train a Gradient Boosting Machine classifier on the selected features
gbm_clf = GradientBoostingClassifier(random_state=42)
gbm_clf.fit(X_train_selected, y_train_resampled)

# Make predictions on the test set using the model with selected features
y_pred = gbm_clf.predict(X_test_selected)

# Calculate accuracy, precision, recall, f1-score, and support
accuracy = accuracy_score(y_test, y_pred)
precision, recall, f1_score, support = precision_recall_fscore_support(y_test, y_pred, average='binary')

# Print the evaluation metrics
print("Accuracy (GBM with Mutual Information Gain + SMOTE):", accuracy)
print("Precision (GBM with Mutual Information Gain + SMOTE):", precision)
print("Recall (GBM with Mutual Information Gain + SMOTE):", recall)
print("F1-Score (GBM with Mutual Information Gain + SMOTE):", f1_score)
print("Support (GBM with Mutual Information Gain + SMOTE):", support)
