import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_predict
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score, f1_score,
    roc_curve, precision_recall_curve, precision_score, recall_score
)
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

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

def evaluate_top_features(input_file, output_dir, model_name):
    try:
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)  

        # Load data
        top10_df = pd.read_csv(input_file)

        # Extract features and target
        top10_df_array = top10_df.columns[:-1]
        y = top10_df['condition']

        # Cross-validation setup
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)

        # Retrieve model and parameter grid
        model = classifiers[model_name]
        param_grid = param_grids.get(model_name, {})

        # Initialize storage for metrics and curves
        performance_metrics = []
        roc_curves = []
        pr_curves = []

        # Loop over top features from 10 to 1
        for n_features in range(10, 0, -1):
            print(f"\nEvaluating model with top {n_features} features...")

            # Select top n features
            selected_features = list(top10_df_array[:n_features])
            X = top10_df[selected_features]

            # Perform hyperparameter tuning with GridSearchCV
            grid_search = GridSearchCV(model, param_grid, cv=cv, scoring='roc_auc', n_jobs=-1)
            grid_search.fit(X, y)

            # Use the best model
            final_model = grid_search.best_estimator_

            # Evaluate using cross-validation
            y_pred_proba = cross_val_predict(final_model, X, y, cv=cv, method='predict_proba')[:, 1]
            y_pred = (y_pred_proba > 0.5).astype(int)

            # Calculate evaluation metrics
            roc_auc = roc_auc_score(y, y_pred_proba)
            pr_auc = average_precision_score(y, y_pred_proba)
            f1 = f1_score(y, y_pred)
            accuracy = accuracy_score(y, y_pred)
            precision = precision_score(y, y_pred)
            recall = recall_score(y, y_pred)

            # Compute and store ROC Curve
            fpr, tpr, _ = roc_curve(y, y_pred_proba)
            roc_curves.append((fpr, tpr, roc_auc, n_features))

            # Compute and store Precision-Recall Curve
            precision_curve, recall_curve, _ = precision_recall_curve(y, y_pred_proba)
            pr_curves.append((precision_curve, recall_curve, pr_auc, n_features))

            # Save performance metrics
            performance_metrics.append({
                'Number of Features': n_features,
                'AUPRC': pr_auc,
                'AUROC': roc_auc,
                'Precision': precision,
                'Recall': recall,
                'F1 Score': f1,
                'Accuracy': accuracy
            })

        # Save performance metrics to CSV
        metrics_df = pd.DataFrame(performance_metrics)
        metrics_csv_path = os.path.join(output_dir, f"{model_name}_feature_performance_metrics.csv")
        metrics_df.to_csv(metrics_csv_path, index=False)

        # --- Generate and Save the Performance Metrics Plot ---
        plt.figure(figsize=(12, 6))

        # Plot each metric over the number of features
        plt.plot(metrics_df['Number of Features'], metrics_df['Accuracy'], label='Accuracy', marker='o', color='blue')
        plt.plot(metrics_df['Number of Features'], metrics_df['AUROC'], label='AUROC', marker='o', color='green')
        plt.plot(metrics_df['Number of Features'], metrics_df['AUPRC'], label='AUPRC', marker='o', color='orange')

        # Invert x-axis for descending number of features
        plt.gca().invert_xaxis()

        # Add labels, title, legend, and grid
        plt.title(f"Performance Metrics by Number of Features ({model_name})")
        plt.xlabel("Number of Features")
        plt.ylabel("Score")
        plt.legend(loc="best")
        plt.grid()

        # Save the updated performance metrics plot
        plot_path = os.path.join(output_dir, f"{model_name}_feature_performance_plot.png")
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        print("Updated performance metrics plot saved to:", plot_path)

        # --- Additional Code for Selecting Best Features by AUPRC ---
        if not pr_curves:
            raise ValueError("pr_curves is empty. Ensure the Precision-Recall curves are computed before selecting features.")

        # Select the top model based on AUPRC
        best_pr_curve = max(pr_curves, key=lambda x: x[2])  # Get the curve with the highest AUPRC
        best_pr_n_features = best_pr_curve[3]  # Number of features for the best AUPRC

        print(f"Best model based on AUPRC uses {best_pr_n_features} features.")

        # Ensure selected_features contains valid feature names
        selected_features = list(top10_df_array[:best_pr_n_features])
        missing_features = [feature for feature in selected_features if feature not in top10_df.columns]

        if missing_features:
            raise ValueError(f"The following features are missing in top10_df: {missing_features}")

        # Create the final DataFrame with selected features and 'condition'
        final_df = top10_df[selected_features + ['condition']]

        # Save the final DataFrame
        final_df_path = os.path.join(output_dir, 'final_selected_features_auprc.csv')
        final_df.to_csv(final_df_path, index=False)

        print("Final selected features saved to:", final_df_path)

        return {
            "message": "Feature evaluation completed successfully.",
            "metrics_file": metrics_csv_path,
            "plot_file": plot_path,
            "final_selected_features": final_df_path
        }

    except Exception as e:
        return {"message": "Error during feature evaluation.", "error": str(e)}

if __name__ == "__main__":
    import sys
    input_file = sys.argv[1]
    output_dir = sys.argv[2]
    model_name = sys.argv[3]
    result = evaluate_top_features(input_file, output_dir, model_name)
    print(result)
