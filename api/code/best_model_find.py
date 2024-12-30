import pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, accuracy_score,
    precision_score, recall_score, roc_curve, precision_recall_curve
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
import os
import sys
import numpy as np



# Define classifiers and hyperparameter grids
classifiers = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=123),
    'Extra Trees': ExtraTreesClassifier(random_state=123),
    'Random Forest': RandomForestClassifier(random_state=123),
    'XGBoost': XGBClassifier(eval_metric='logloss', random_state=123),
    'Gradient Boosting': GradientBoostingClassifier(random_state=123),
    'SVM': SVC(probability=True, random_state=123),
    'LDA': LinearDiscriminantAnalysis(),
    'QDA': QuadraticDiscriminantAnalysis(),
    'Naive Bayes': GaussianNB(),
    'K Neighbors': KNeighborsClassifier(),
    'AdaBoost': AdaBoostClassifier(algorithm='SAMME', random_state=123)
}

param_grids = {
    'Logistic Regression': {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'solver': ['liblinear', 'saga']
    },
    'Extra Trees': {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10]
    },
    'Random Forest': {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10]
    },
    'LightGBM': {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [-1, 10, 20]
    },
    'XGBoost': {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    },
    'Gradient Boosting': {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    },
    'K Neighbors': {
        'n_neighbors': [3, 5, 7, 9],
        'weights': ['uniform', 'distance']
    },
    'AdaBoost': {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 1.0]
    }
}


def benchmark_models(input_file, output_dir):
    try:
        # Load dataset
        df = pd.read_csv(input_file)
        X = df.drop(columns=['condition'])
        y = df['condition']

        os.makedirs(output_dir, exist_ok=True)

        # Stratified Cross-Validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)

        metrics = []

        # Benchmark each classifier
        for name, model in classifiers.items():
            print(f"Evaluating {name}...")

            # Cross-validation predictions
            y_pred_proba = cross_val_predict(model, X, y, cv=cv, method='predict_proba', n_jobs=-1)[:, 1]
            auc = roc_auc_score(y, y_pred_proba)
            auprc = average_precision_score(y, y_pred_proba)

            # Hyperparameter tuning if applicable
            if name in param_grids:
                grid_search = GridSearchCV(model, param_grids[name], cv=cv, scoring='roc_auc', n_jobs=-1)
                grid_search.fit(X, y)
                model = grid_search.best_estimator_

                # Recalculate metrics after tuning
                y_pred_proba = cross_val_predict(model, X, y, cv=cv, method='predict_proba', n_jobs=-1)[:, 1]
                auc = roc_auc_score(y, y_pred_proba)
                auprc = average_precision_score(y, y_pred_proba)

            # Optimize threshold for F1-Score
            thresholds = [t / 100 for t in range(0, 100)]
            best_f1, best_threshold = max(
                (f1_score(y, (y_pred_proba > t).astype(int)), t) for t in thresholds
            )
            y_pred = (y_pred_proba > best_threshold).astype(int)
            accuracy = accuracy_score(y, y_pred)
            precision_at_threshold = precision_score(y, y_pred)
            recall_at_threshold = recall_score(y, y_pred)

            # Store metrics
            metrics.append({
                'Model': name,
                'AUPRC': auprc,
                'AUROC': auc,
                'Precision': precision_at_threshold,
                'Recall': recall_at_threshold,
                'F1-Score': best_f1,
                'Accuracy': accuracy
            })

        # Convert metrics to DataFrame
        metrics_df = pd.DataFrame(metrics)
        metrics_df = metrics_df.sort_values(by=['AUPRC'], ascending=False).reset_index(drop=True)

        # Save metrics
        metrics_path = os.path.join(output_dir, "model_benchmarking_results.csv")
        metrics_df.to_csv(metrics_path, index=False)

        # Determine the best model
        best_model = metrics_df.iloc[0]['Model']

        return f"{metrics_path}"

    except Exception as e:
        return f"Error: {str(e)}"


if __name__ == "__main__":
    import sys
    input_file = sys.argv[1]
    output_dir = sys.argv[2]
    print(benchmark_models(input_file, output_dir))
