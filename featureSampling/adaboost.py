import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import AdaBoostClassifier
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

# Train the AdaBoost model on the selected features
adaboost_clf = AdaBoostClassifier(random_state=42)
adaboost_clf.fit(X_train_selected_pearson, y_train_resampled)

# Make predictions on the test set using the model with selected features
y_pred = adaboost_clf.predict(X_test_selected_pearson)

# Calculate accuracy, precision, recall, F1-score, and support
accuracy = accuracy_score(y_test, y_pred)
precision, recall, f1_score, support = precision_recall_fscore_support(y_test, y_pred, average='binary')

# Print the evaluation metrics
print("Accuracy (AdaBoost with Pearson + SMOTE):", accuracy)
print("Precision (AdaBoost with Pearson + SMOTE):", precision)
print("Recall (AdaBoost with Pearson + SMOTE):", recall)
print("F1-Score (AdaBoost with Pearson + SMOTE):", f1_score)
print("Support (AdaBoost with Pearson + SMOTE):", support)
