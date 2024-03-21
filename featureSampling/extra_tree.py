import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_classif
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
X_train_selected = selector.fit_transform(X_train_resampled, y_train_resampled)
X_test_selected = selector.transform(X_test)

# Train the Extra Trees model on the selected features using 10-fold cross-validation
et_clf = ExtraTreesClassifier(random_state=42)
cv_scores_et = cross_val_score(et_clf, X_train_selected, y_train_resampled, cv=10, scoring='accuracy')

# Print cross-validation scores
print("Cross-Validation Scores (Extra Trees with SMOTE and Pearson):", cv_scores_et)
print("Mean Cross-Validation Accuracy (Extra Trees with SMOTE and Pearson):", cv_scores_et.mean())

# Fit the Extra Trees model on the full training set
et_clf.fit(X_train_selected, y_train_resampled)

# Make predictions on the test set using the model with selected features
y_pred_selected_et = et_clf.predict(X_test_selected)

# Calculate accuracy, precision, recall, F1-score, and support
accuracy_et = accuracy_score(y_test, y_pred_selected_et)
precision_et, recall_et, f1_score_et, support_et = precision_recall_fscore_support(y_test, y_pred_selected_et, average='binary')

# Print the evaluation metrics
print("Accuracy (Extra Trees with SMOTE and Pearson):", accuracy_et)
print("Precision (Extra Trees with SMOTE and Pearson):", precision_et)
print("Recall (Extra Trees with SMOTE and Pearson):", recall_et)
print("F1-Score (Extra Trees with SMOTE and Pearson):", f1_score_et)
print("Support (Extra Trees with SMOTE and Pearson):", support_et)
