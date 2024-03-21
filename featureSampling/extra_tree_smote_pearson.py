import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectKBest, f_classif

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

# Define the base classifiers
clf1 = LogisticRegression(random_state=42)
clf2 = DecisionTreeClassifier(random_state=42)
clf3 = SVC(random_state=42)

# Create the Voting Classifier
voting_clf = VotingClassifier(estimators=[('lr', clf1), ('dt', clf2), ('svm', clf3)], voting='hard')

# Train the Voting Classifier
voting_clf.fit(X_train_selected, y_train_resampled)

# Make predictions on the test set
y_pred = voting_clf.predict(X_test_selected)

# Calculate accuracy, precision, recall, F1-score
accuracy = accuracy_score(y_test, y_pred)
precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')

print("Accuracy (Voting Classifier with SMOTE and Pearson):", accuracy)
print("Precision (Voting Classifier with SMOTE and Pearson):", precision)
print("Recall (Voting Classifier with SMOTE and Pearson):", recall)
print("F1-Score (Voting Classifier with SMOTE and Pearson):", f1_score)
