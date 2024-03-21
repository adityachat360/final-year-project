import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, roc_auc_score
from xgboost import XGBClassifier

# Set the seed for reproducibility
np.random.seed(2021)

# Load the data from CSV file
data = pd.read_csv('thyroid_clean.csv')
  # Replace 'path/to/your/file.csv' with the actual path to your CSV file

# Convert 'mal' column to character
data['mal'] = data['mal'].astype(str)

# Function to calculate confusion matrix and ROC AUC
def calculate_metrics(y_true, y_pred, y_scores):
    cm = confusion_matrix(y_true, y_pred)
    auc = roc_auc_score(y_true, y_scores)
    acc = np.sum(np.diag(cm)) / np.sum(cm)
    sen = cm[1, 1] / np.sum(cm[1, :])
    spec = cm[0, 0] / np.sum(cm[0, :])
    prec = cm[1, 1] / np.sum(cm[:, 1])
    return acc, auc, sen, spec, prec

# Function to perform bootstrap sampling and train XGBoost model
def perform_bootstrap(data, boost_count=1000):
    acc_list, auroc_list, sen_list, spec_list, prec_list = [], [], [], [], []

    for i in range(boost_count):
        print(i)
        indices = np.random.choice(data['id'].unique(), size=len(data['id'].unique()), replace=True)
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=2021)

        results = pd.DataFrame()

        for train_index, test_index in skf.split(indices, indices):
            train_ids, test_ids = indices[train_index], indices[test_index]
            train = data[data['id'].isin(train_ids)]
            test = data[data['id'].isin(test_ids)]

            # Resample with replacement on the training set
            bootstrap_index = np.random.choice(len(train), size=len(train), replace=True)
            train = train.iloc[bootstrap_index]

            # XGBoost model
            model = XGBClassifier()
            model.fit(train.drop(['id', 'mal'], axis=1), train['mal'])
            pred_prob = model.predict_proba(test.drop(['id', 'mal'], axis=1))[:, 1]

            results = pd.concat([results, pd.DataFrame({'pred_prob': pred_prob}, index=test.index)])

        data['pred_prob'] = results['pred_prob']
        data['mal'] = data['mal'].astype(int)
        data['pred_mal'] = (data['pred_prob'] >= 0.5).astype(int)

        acc, auroc, sen, spec, prec = calculate_metrics(data['mal'], data['pred_mal'], data['pred_prob'])
        acc_list.append(acc)
        auroc_list.append(auroc)
        sen_list.append(sen)
        spec_list.append(spec)
        prec_list.append(prec)

    return acc_list, auroc_list, sen_list, spec_list, prec_list

# Perform bootstrap and calculate statistics
result_bootstrap = perform_bootstrap(data)

# Print statistics
print("Accuracy: ", np.percentile(result_bootstrap[0], [2.5, 97.5]))
print("Mean Accuracy: ", np.mean(result_bootstrap[0]))
print("AUROC: ", np.percentile(result_bootstrap[1], [2.5, 97.5]))
print("Mean AUROC: ", np.mean(result_bootstrap[1]))
print("Sensitivity: ", np.percentile(result_bootstrap[2], [2.5, 97.5]))
print("Mean Sensitivity: ", np.mean(result_bootstrap[2]))
print("Specificity: ", np.percentile(result_bootstrap[3], [2.5, 97.5]))
print("Mean Specificity: ", np.mean(result_bootstrap[3]))
print("Precision: ", np.percentile(result_bootstrap[4], [2.5, 97.5]))
print("Mean Precision: ", np.mean(result_bootstrap[4]))