

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
import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score


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

        # Initialize Stratified Cross-Validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)

        # Initialize storage for models and metrics
        best_models = {}
        metrics = []
        roc_curves = {}
        pr_curves = {}

        for name, model in classifiers.items():
            print(f"Tuning {name}...")

            # Cross-validation predictions for original model
            y_pred_proba_original = cross_val_predict(
                model, X, y, cv=cv, method='predict_proba', n_jobs=-1
            )[:, 1]
            original_auc = roc_auc_score(y, y_pred_proba_original)
            original_auprc = average_precision_score(y, y_pred_proba_original)

            # Evaluate tuned model if hyperparameter grid exists
            if name in param_grids:
                grid_search = GridSearchCV(
                    model, param_grids[name], cv=cv, scoring='roc_auc', n_jobs=-1
                )
                grid_search.fit(X, y)
                tuned_model = grid_search.best_estimator_

                y_pred_proba_tuned = cross_val_predict(
                    tuned_model, X, y, cv=cv, method='predict_proba', n_jobs=-1
                )[:, 1]

                tuned_auc = roc_auc_score(y, y_pred_proba_tuned)
                tuned_auprc = average_precision_score(y, y_pred_proba_tuned)

                # Choose the better model
                if tuned_auc > original_auc:
                    best_model = tuned_model
                else:
                    best_model = model
            else:
                best_model = model

            # Save model parameters for JSON serialization
            best_models[name] = {
                "model_name": type(best_model).__name__,
                "parameters": best_model.get_params()
            }

            # Store metrics
            metrics.append({
                'Model': name,
                'AUPRC': original_auprc,
                'AUROC': original_auc,
            })

        # Convert metrics to DataFrame
        metrics_df = pd.DataFrame(metrics)
        metrics_df = metrics_df.sort_values(by=['AUPRC'], ascending=False).reset_index(drop=True)

        # Save metrics
        metrics_path = os.path.join(output_dir, "model_benchmarking_results.csv")
        metrics_df.to_csv(metrics_path, index=False)

        # Save best_models in proper JSON format
        best_models_path = os.path.join(output_dir, "best_models.json")
        with open(best_models_path, "w") as json_file:
            json.dump(best_models, json_file, indent=4)  # Save as a JSON file

        # Sort models by AUPRC
        sorted_models = sorted(best_models.items(), key=lambda x: metrics_df.loc[metrics_df['Model'] == x[0], 'AUPRC'].values[0], reverse=True)

        # Create a figure with 2 subplots
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # --- AUPRC Curves ---
        for name, model in sorted_models:
            # Retrieve predictions
            y_pred_proba_cv = cross_val_predict(model, X, y, cv=cv, method='predict_proba', n_jobs=-1)[:, 1]
            precision, recall, _ = precision_recall_curve(y, y_pred_proba_cv)
            auprc = average_precision_score(y, y_pred_proba_cv)

            # Plot Precision-Recall curve
            axes[0].plot(recall, precision, lw=1.75, label=f'{name} (AUPRC = {auprc:.2f})')

        axes[0].set_title('AUPRC Curves')
        axes[0].set_xlabel('Recall')
        axes[0].set_ylabel('Precision')
        axes[0].set_xlim([0, 1])
        axes[0].set_ylim([0, 1.05])
        axes[0].legend(loc='lower left', fontsize=9, frameon=False)


        # --- AUROC Curves ---
        for name, model in sorted_models:
            # Retrieve predictions
            y_pred_proba_cv = cross_val_predict(model, X, y, cv=cv, method='predict_proba', n_jobs=-1)[:, 1]
            fpr, tpr, _ = roc_curve(y, y_pred_proba_cv)
            auc = roc_auc_score(y, y_pred_proba_cv)

            # Plot ROC curve
            axes[1].plot(fpr, tpr, lw=1.75, label=f'{name} (AUROC = {auc:.2f})')

        axes[1].plot([0, 1], [0, 1], 'k--', label='Random Chance')
        axes[1].set_title('AUROC Curves')
        axes[1].set_xlabel('False Positive Rate')
        axes[1].set_ylabel('True Positive Rate')
        axes[1].set_xlim([0, 1])
        axes[1].set_ylim([0, 1.05])
        axes[1].legend(loc='lower right', fontsize=9, frameon=False)


        # Add a main title for the figure
        fig.suptitle('Model Benchmarking: AUPRC and AUROC', fontsize=16, y=1)

        # Adjust layout for landscape orientation
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        # Save the figure
         # Save the figure
        png_path = os.path.join(output_dir, 'model_benchmarking_curves.png')
        pdf_path = os.path.join(output_dir, 'model_benchmarking_curves.pdf')
        fig.savefig(png_path, dpi=300, bbox_inches='tight')
        fig.savefig(pdf_path, dpi=300, bbox_inches='tight')

        # Show the combined figure
        plt.show()


        return {
            "metrics_path": metrics_path,
            "best_models_path": best_models_path
        }

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    import sys
    input_file = sys.argv[1]
    output_dir = sys.argv[2]
    print(benchmark_models(input_file, output_dir))