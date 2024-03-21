import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset (replace 'thyroid_clean.csv' with the actual path)
data = pd.read_csv('thyroid_clean.csv')

# Assuming 'mal' is the column indicating the presence of thyroid cancer
X = data.drop(['mal', 'id'], axis=1)  # Features, 'id' is assumed to be an identifier column
y = data['mal']  # Target variable

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature selection using SelectKBest with ANOVA F-statistic
k_best = SelectKBest(f_classif, k=10)  # Select top 10 features
X_train_selected_knn = k_best.fit_transform(X_train, y_train)
X_test_selected_knn = k_best.transform(X_test)

# Initialize a k-Nearest Neighbors model
knn_clf = KNeighborsClassifier()

# Train the KNN model on the selected features using k-fold cross-validation
cv_scores_knn = cross_val_score(knn_clf, X_train_selected_knn, y_train, cv=5, scoring='accuracy')

# Print cross-validation scores
print("Cross-Validation Scores (KNN):", cv_scores_knn)
print("Mean Cross-Validation Accuracy (KNN):", cv_scores_knn.mean())

# Fit the KNN model on the full training set
knn_clf.fit(X_train_selected_knn, y_train)

# Make predictions on the test set using the model with selected features
y_pred_selected_knn = knn_clf.predict(X_test_selected_knn)

# Calculate accuracy, precision, recall, F1-score, and support
accuracy_knn = accuracy_score(y_test, y_pred_selected_knn)
precision_knn, recall_knn, f1_score_knn, support_knn = precision_recall_fscore_support(y_test, y_pred_selected_knn, average='binary')

# Print the evaluation metrics
print("Accuracy (KNN):", accuracy_knn)
print("Precision (KNN):", precision_knn)
print("Recall (KNN):", recall_knn)
print("F1-Score (KNN):", f1_score_knn)
print("Support (KNN):", support_knn)

# Generate ROC Curve
fpr_knn, tpr_knn, thresholds_knn = roc_curve(y_test, knn_clf.predict_proba(X_test_selected_knn)[:, 1])
roc_auc_knn = auc(fpr_knn, tpr_knn)

plt.figure(figsize=(10, 6))
plt.plot(fpr_knn, tpr_knn, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc_knn))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve for KNN')
plt.legend(loc="lower right")
plt.show()

# Generate Confusion Matrix
cm_knn = confusion_matrix(y_test, y_pred_selected_knn)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_knn, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix for KNN')
plt.show()
