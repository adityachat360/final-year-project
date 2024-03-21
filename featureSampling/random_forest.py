import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
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

# Feature selection using Pearson correlation
selector = SelectKBest(f_classif, k=10)  # Select top 10 features (adjust k as needed)
X_train_selected_pearson = selector.fit_transform(X_train_resampled, y_train_resampled)
X_test_selected_pearson = selector.transform(X_test)

# Train the Random Forest model on the selected features using 10-fold cross-validation
rf_clf = RandomForestClassifier(random_state=42)
cv_scores_pearson = cross_val_score(rf_clf, X_train_selected_pearson, y_train_resampled, cv=10, scoring='accuracy')

# Print cross-validation scores
print("Cross-Validation Scores (Random Forest with Pearson + SMOTE):", cv_scores_pearson)
print("Mean Cross-Validation Accuracy (Random Forest with Pearson + SMOTE):", cv_scores_pearson.mean())

# Fit the Random Forest model on the full training set
rf_clf.fit(X_train_selected_pearson, y_train_resampled)

# Make predictions on the test set using the model with selected features
y_pred_selected_pearson = rf_clf.predict(X_test_selected_pearson)

# Calculate accuracy, precision, recall, F1-score, and support
accuracy_pearson = accuracy_score(y_test, y_pred_selected_pearson)
precision_pearson, recall_pearson, f1_score_pearson, support_pearson = precision_recall_fscore_support(y_test, y_pred_selected_pearson, average='binary')

# Print the evaluation metrics
print("Accuracy (Random Forest with Pearson + SMOTE):", accuracy_pearson)
print("Precision (Random Forest with Pearson + SMOTE):", precision_pearson)
print("Recall (Random Forest with Pearson + SMOTE):", recall_pearson)
print("F1-Score (Random Forest with Pearson + SMOTE):", f1_score_pearson)
print("Support (Random Forest with Pearson + SMOTE):", support_pearson)
