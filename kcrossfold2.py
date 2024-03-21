import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_curve, auc
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from itertools import cycle

# Load your dataset (replace 'thyroid_clean.csv' with the actual path)
data = pd.read_csv('thyroid_clean.csv')

# Assuming 'mal' is the column indicating the presence of thyroid cancer
X = data.drop(['mal', 'id'], axis=1)  # Features, 'id' is assumed to be an identifier column
y = data['mal']  # Target variable

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize a Random Forest Classifier
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the Random Forest model
rf_clf.fit(X_train, y_train)

# Make predictions on the test set using Random Forest
y_pred_rf = rf_clf.predict(X_test)

# Evaluate the Random Forest model
accuracy_rf = accuracy_score(y_test, y_pred_rf)
classification_report_rf = classification_report(y_test, y_pred_rf)

# Print the results for Random Forest
print("Random Forest Results:")
print(f"Accuracy: {accuracy_rf:.4f}")
print("Classification Report:\n", classification_report_rf)

# Initialize an AdaBoost Classifier with a base estimator (e.g., Decision Tree)
ada_clf = AdaBoostClassifier(base_estimator=RandomForestClassifier(n_estimators=50), n_estimators=100, random_state=42)

# Train the AdaBoost model
ada_clf.fit(X_train, y_train)

# Make predictions on the test set using AdaBoost
y_pred_ada = ada_clf.predict(X_test)

# Evaluate the AdaBoost model
accuracy_ada = accuracy_score(y_test, y_pred_ada)
classification_report_ada = classification_report(y_test, y_pred_ada)

# Print the results for AdaBoost
print("\nAdaBoost Results:")
print(f"Accuracy: {accuracy_ada:.4f}")
print("Classification Report:\n", classification_report_ada)

# Initialize StratifiedKFold for k-fold cross-validation with n_splits=10
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Cross-validation for Random Forest
rf_cv_scores = cross_val_score(rf_clf, X, y, cv=kf, scoring='accuracy')
print("\nRandom Forest Cross-Validation Scores:")
print(f"Mean Accuracy: {rf_cv_scores.mean():.4f}")
print("Individual Fold Accuracies:", rf_cv_scores)

# Cross-validation for AdaBoost
ada_cv_scores = cross_val_score(ada_clf, X, y, cv=kf, scoring='accuracy')
print("\nAdaBoost Cross-Validation Scores:")
print(f"Mean Accuracy: {ada_cv_scores.mean():.4f}")
print("Individual Fold Accuracies:", ada_cv_scores)

# Plot and save AUROC curves
def plot_roc_curve(fpr, tpr, auc, label):
    plt.plot(fpr, tpr, label=f'{label} (AUC = {auc:.2f})')

# Plotting for Random Forest
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_clf.predict_proba(X_test)[:, 1])
roc_auc_rf = auc(fpr_rf, tpr_rf)
plot_roc_curve(fpr_rf, tpr_rf, roc_auc_rf, 'Random Forest')

# Plotting for AdaBoost
fpr_ada, tpr_ada, _ = roc_curve(y_test, ada_clf.predict_proba(X_test)[:, 1])
roc_auc_ada = auc(fpr_ada, tpr_ada)
plot_roc_curve(fpr_ada, tpr_ada, roc_auc_ada, 'AdaBoost')

# Set plot properties
plt.plot([0, 1], [0, 1], linestyle='--', color='grey', label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.grid(True)

# Save the plot
plt.savefig('roc_curve.png')
plt.show()