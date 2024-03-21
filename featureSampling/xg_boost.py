import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from xgboost import XGBClassifier
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

# Train the XGBoost model on the selected features using 10-fold cross-validation
xgb_clf = XGBClassifier(random_state=42)
cv_scores_xgb = cross_val_score(xgb_clf, X_train_selected, y_train_resampled, cv=10, scoring='accuracy')

# Print cross-validation scores
print("Cross-Validation Scores (XGBoost with SMOTE and Pearson):", cv_scores_xgb)
print("Mean Cross-Validation Accuracy (XGBoost with SMOTE and Pearson):", cv_scores_xgb.mean())

# Fit the XGBoost model on the full training set
xgb_clf.fit(X_train_selected, y_train_resampled)

# Make predictions on the test set using the model with selected features
y_pred_selected_xgb = xgb_clf.predict(X_test_selected)

# Calculate accuracy, precision, recall, F1-score, and support
accuracy_xgb = accuracy_score(y_test, y_pred_selected_xgb)
precision_xgb, recall_xgb, f1_score_xgb, support_xgb = precision_recall_fscore_support(y_test, y_pred_selected_xgb, average='binary')

# Print the evaluation metrics
print("Accuracy (XGBoost with SMOTE and Pearson):", accuracy_xgb)
print("Precision (XGBoost with SMOTE and Pearson):", precision_xgb)
print("Recall (XGBoost with SMOTE and Pearson):", recall_xgb)
print("F1-Score (XGBoost with SMOTE and Pearson):", f1_score_xgb)
print("Support (XGBoost with SMOTE and Pearson):", support_xgb)
