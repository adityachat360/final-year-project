import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
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

# Train the Decision Tree model on the selected features using 10-fold cross-validation
dt_clf = DecisionTreeClassifier(random_state=42)
cv_scores_dt = cross_val_score(dt_clf, X_train_selected, y_train_resampled, cv=10, scoring='accuracy')

# Print cross-validation scores
print("Cross-Validation Scores (Decision Tree with SMOTE and Pearson):", cv_scores_dt)
print("Mean Cross-Validation Accuracy (Decision Tree with SMOTE and Pearson):", cv_scores_dt.mean())

# Fit the Decision Tree model on the full training set
dt_clf.fit(X_train_selected, y_train_resampled)

# Make predictions on the test set using the model with selected features
y_pred_selected_dt = dt_clf.predict(X_test_selected)

# Calculate accuracy, precision, recall, F1-score, and support
accuracy_dt = accuracy_score(y_test, y_pred_selected_dt)
precision_dt, recall_dt, f1_score_dt, support_dt = precision_recall_fscore_support(y_test, y_pred_selected_dt, average='binary')

# Print the evaluation metrics
print("Accuracy (Decision Tree with SMOTE and Pearson):", accuracy_dt)
print("Precision (Decision Tree with SMOTE and Pearson):", precision_dt)
print("Recall (Decision Tree with SMOTE and Pearson):", recall_dt)
print("F1-Score (Decision Tree with SMOTE and Pearson):", f1_score_dt)
print("Support (Decision Tree with SMOTE and Pearson):", support_dt)
