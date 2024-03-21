import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_curve, roc_auc_score
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

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

# Train the Gradient Boosting model on the selected features using 10-fold cross-validation
gbm_clf = GradientBoostingClassifier(random_state=42)
cv_scores_pearson = cross_val_score(gbm_clf, X_train_selected_pearson, y_train_resampled, cv=10, scoring='accuracy')

# Print cross-validation scores
print("Cross-Validation Scores (GBM with Pearson + SMOTE):", cv_scores_pearson)
print("Mean Cross-Validation Accuracy (GBM with Pearson + SMOTE):", cv_scores_pearson.mean())

# Fit the Gradient Boosting model on the full training set
gbm_clf.fit(X_train_selected_pearson, y_train_resampled)

# Make predictions on the test set using the model with selected features
y_pred_selected_pearson = gbm_clf.predict(X_test_selected_pearson)

# Calculate accuracy, precision, recall, F1-score, and support
accuracy_pearson = accuracy_score(y_test, y_pred_selected_pearson)
precision_pearson, recall_pearson, f1_score_pearson, support_pearson = precision_recall_fscore_support(y_test, y_pred_selected_pearson, average='binary')

# Print the evaluation metrics
print("Accuracy (GBM with Pearson + SMOTE):", accuracy_pearson)
print("Precision (GBM with Pearson + SMOTE):", precision_pearson)
print("Recall (GBM with Pearson + SMOTE):", recall_pearson)
print("F1-Score (GBM with Pearson + SMOTE):", f1_score_pearson)
print("Support (GBM with Pearson + SMOTE):", support_pearson)

# Calculate predicted probabilities for the positive class
y_pred_prob_pearson = gbm_clf.predict_proba(X_test_selected_pearson)[:, 1]

# Calculate ROC curve
fpr, tpr, _ = roc_curve(y_test, y_pred_prob_pearson)

# Calculate AUC score
auc_score = roc_auc_score(y_test, y_pred_prob_pearson)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label='GBM with Pearson + SMOTE (AUC = {:.2f})'.format(auc_score))
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()
