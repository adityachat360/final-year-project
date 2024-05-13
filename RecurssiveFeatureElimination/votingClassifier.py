import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import RFE

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

# Initialize Voting Classifier with soft voting
rf_clf = RandomForestClassifier(random_state=42)
gb_clf = GradientBoostingClassifier(random_state=42)
lr_clf = LogisticRegression(random_state=42)

estimators = [('rf', rf_clf), ('gb', gb_clf), ('lr', lr_clf)]
voting_clf = VotingClassifier(estimators=estimators, voting='soft')

# Fit RFE for each estimator in the Voting Classifier
for name, estimator in voting_clf.named_estimators_.items():
    rfe = RFE(estimator=estimator, n_features_to_select=10, step=1)
    rfe.fit(X_train_resampled, y_train_resampled)
    selected_features = rfe.support_
    X_train_resampled = X_train_resampled[:, selected_features]
    X_test = X_test[:, selected_features]

# Train the Voting Classifier on the selected features
voting_clf.fit(X_train_resampled, y_train_resampled)

# Make predictions on the test set using the model with selected features
y_pred = voting_clf.predict(X_test)

# Calculate accuracy, precision, recall, f1-score, and support
accuracy = accuracy_score(y_test, y_pred)
precision, recall, f1_score, support = precision_recall_fscore_support(y_test, y_pred, average='binary')

# Print the evaluation metrics
print("Accuracy (Voting Classifier with RFE + SMOTE):", accuracy)
print("Precision (Voting Classifier with RFE + SMOTE):", precision)
print("Recall (Voting Classifier with RFE + SMOTE):", recall)
print("F1-Score (Voting Classifier with RFE + SMOTE):", f1_score)
print("Support (Voting Classifier with RFE + SMOTE):", support)
