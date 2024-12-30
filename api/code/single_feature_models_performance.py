from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, average_precision_score,
    f1_score, accuracy_score, roc_curve, precision_score, recall_score
)
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_predict
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier
)

# Define a global variable for the selected model name (set by API)
selected_model_name = None

# Define classifiers and parameter grids
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
    'Logistic Regression': {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'solver': ['liblinear']},
    'Extra Trees': {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20], 'min_samples_split': [2, 5, 10]},
    'Random Forest': {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20], 'min_samples_split': [2, 5, 10]},
    'XGBoost': {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.2], 'max_depth': [3, 5, 7]},
    'Gradient Boosting': {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.2], 'max_depth': [3, 5, 7]},
    'K Neighbors': {'n_neighbors': [3, 5, 7, 9], 'weights': ['uniform', 'distance']},
    'AdaBoost': {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 1.0]}
}

def evaluate_single_feature_models(input_file, output_dir, model_name):
    try:
        global selected_model_name

        selected_model_name = model_name

        # Validate global model name
        if not selected_model_name:
            raise ValueError("No model name selected. Please set the global model name first.")

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Load the dataset
        top10_df = pd.read_csv(input_file)

        # Extract features and target
        X = top10_df.drop(columns=['condition'])
        y = top10_df['condition']

        # Cross-validation setup
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)

        # Retrieve model and parameter grid
        model = classifiers[selected_model_name]
        param_grid = param_grids.get(selected_model_name, {})

        # Store results for metrics
        metrics_scores = []

        # Loop through top features for individual modeling and tuning
        for feature in X.columns:
            print(f"Processing feature: {feature}")

            # Prepare single feature data
            X_single = X[[feature]]

            # Perform GridSearchCV for hyperparameter tuning
            grid_search = GridSearchCV(model, param_grid, cv=cv, scoring='roc_auc', n_jobs=-1)
            grid_search.fit(X_single, y)

            # Get the best model
            best_model = grid_search.best_estimator_

            # Perform cross-validation to get prediction probabilities
            y_pred_proba = cross_val_predict(
                best_model, X_single, y, cv=cv, method='predict_proba', n_jobs=-1
            )[:, 1]

            # Perform cross-validation to get predictions
            y_pred = cross_val_predict(best_model, X_single, y, cv=cv, method='predict', n_jobs=-1)

            # Calculate metrics
            roc_auc = roc_auc_score(y, y_pred_proba)
            precision, recall, _ = precision_recall_curve(y, y_pred_proba)
            auprc = average_precision_score(y, y_pred_proba)
            f1 = f1_score(y, y_pred)
            accuracy = accuracy_score(y, y_pred)

            # Calculate Precision and Recall at the default threshold (0.5)
            precision_at_threshold = precision_score(y, y_pred)
            recall_at_threshold = recall_score(y, y_pred)

            # Store metrics
            metrics_scores.append({
                'Feature': feature,
                'AUPRC': auprc,
                'AUROC': roc_auc,
                'Precision': precision_at_threshold,
                'Recall': recall_at_threshold,
                'F1-Score': f1,
                'Accuracy': accuracy
            })

        # Convert metrics to a DataFrame for analysis
        metrics_df = pd.DataFrame(metrics_scores)

        # Sort features based on the primary metric in descending order
        metrics_df.sort_values(by='AUPRC', ascending=False, inplace=True)

        # Save the sorted results to a CSV file
        metrics_csv_path = os.path.join(output_dir, f"{model_name}_feature_metrics_ranking.csv")
        metrics_df.to_csv(metrics_csv_path, index=False)

        print("Feature metrics saved to:", metrics_csv_path)

        return {
            "message": "Feature evaluation completed successfully.",
            "metrics_file": metrics_csv_path
        }

    except Exception as e:
        return {"message": "Error during feature evaluation.", "error": str(e)}

if __name__ == "__main__":
    import sys
    input_file = sys.argv[1]
    output_dir = sys.argv[2]
    model_name = sys.argv[3]
    result = evaluate_single_feature_models(input_file, output_dir,model_name)
    print(result)
