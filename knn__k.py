import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectKBest, chi2
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

# Feature selection using SelectKBest with chi-squared test
skb = SelectKBest(score_func=chi2, k=5)
X_train_selected = skb.fit_transform(X_train_resampled, y_train_resampled)
X_test_selected = skb.transform(X_test)

# Train the KNN model on the selected features using 10-fold cross-validation
knn = KNeighborsClassifier()
cv_scores_knn = cross_val_score(knn, X_train_selected, y_train_resampled, cv=10, scoring='accuracy')

# Print cross-validation scores
print("Cross-Validation Scores (KNN with SMOTE):", cv_scores_knn)
print("Mean Cross-Validation Accuracy (KNN with SMOTE):", cv_scores_knn.mean())

# Fit the KNN model on the full training set
knn.fit(X_train_selected, y_train_resampled)

# Make predictions on the test set using the model with selected features
y_pred_selected = knn.predict(X_test_selected)

# Calculate accuracy, precision, recall, F1-score, and support
accuracy_knn = accuracy_score(y_test, y_pred_selected)
precision_knn, recall_knn, f1_score_knn, support_knn = precision_recall_fscore_support(y_test, y_pred_selected, average='binary')

# Print the evaluation metrics
print("Accuracy (KNN with SMOTE):", accuracy_knn)
print("Precision (KNN with SMOTE):", precision_knn)
print("Recall (KNN with SMOTE):", recall_knn)
print("F1-Score (KNN with SMOTE):", f1_score_knn)
print("Support (KNN with SMOTE):", support_knn)
