import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

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

# Create a base classifier (Logistic Regression)
base_clf = LogisticRegression()

# Initialize RFE
rfe = RFE(estimator=base_clf, n_features_to_select=10, step=1)

# Fit RFE
rfe.fit(X_train_resampled, y_train_resampled)

# Transform the data
X_train_selected_backward_elimination = rfe.transform(X_train_resampled)
X_test_selected_backward_elimination = rfe.transform(X_test)

# Reshape the data for CNN (assuming X_train_selected_backward_elimination and X_test_selected_backward_elimination are images)
X_train_reshaped = X_train_selected_backward_elimination.reshape(X_train_selected_backward_elimination.shape[0], 28, 28, 1)
X_test_reshaped = X_test_selected_backward_elimination.reshape(X_test_selected_backward_elimination.shape[0], 28, 28, 1)

# Build the CNN model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train_reshaped, y_train_resampled, epochs=10, batch_size=32, validation_data=(X_test_reshaped, y_test))

# Evaluate the model
y_pred_proba = model.predict(X_test_reshaped)
y_pred = (y_pred_proba > 0.5).astype(np.int)

# Calculate accuracy, precision, recall, f1-score, and support
accuracy = accuracy_score(y_test, y_pred)
precision, recall, f1_score, support = precision_recall_fscore_support(y_test, y_pred, average='binary')

# Print the evaluation metrics
print("Accuracy (CNN with Backward Elimination + SMOTE):", accuracy)
print("Precision (CNN with Backward Elimination + SMOTE):", precision)
print("Recall (CNN with Backward Elimination + SMOTE):", recall)
print("F1-Score (CNN with Backward Elimination + SMOTE):", f1_score)
print("Support (CNN with Backward Elimination + SMOTE):", support)
