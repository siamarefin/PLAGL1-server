import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_predict
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
import json

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

def visualize_model_benchmarking(input_file, input_file1, output_dir):
    try:
        # Load benchmarking results
        metrics_df = pd.read_csv(input_file1)

        # Identify best models
        best_models = {
            name: classifiers[name] for name in metrics_df['Model'].values if name in classifiers
        }

        # Load dataset
        df = pd.read_csv(input_file)
        X = df.drop(columns=['condition'])
        y = df['condition']

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Stratified Cross-Validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)

        # Hyperparameter tuning
        tuned_models = {}
        for name, model in best_models.items():
            if name in param_grids:
                grid_search = GridSearchCV(model, param_grids[name], cv=cv, scoring='roc_auc', n_jobs=-1)
                grid_search.fit(X, y)
                tuned_models[name] = grid_search.best_estimator_
            else:
                tuned_models[name] = model

        # Sort models by AUPRC
        sorted_models = sorted(
            tuned_models.items(),
            key=lambda x: metrics_df.loc[metrics_df['Model'] == x[0], 'AUPRC'].values[0],
            reverse=True
        )

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
        png_path = os.path.join(output_dir, 'model_benchmarking_curves.png')
        pdf_path = os.path.join(output_dir, 'model_benchmarking_curves.pdf')
        fig.savefig(png_path, dpi=300, bbox_inches='tight')
        fig.savefig(pdf_path, dpi=300, bbox_inches='tight')

        # Save best models
        best_models_txt = os.path.join(output_dir, 'best_models.txt')
        with open(best_models_txt, "w") as f:
            for name, _ in sorted_models:
                f.write(f"{name}\n")

        return {
            "message": "Visualization completed successfully.",
            "AUPRC_Plot": png_path,
            "AUROC_Plot": pdf_path,
            "Best_Models": best_models_txt
        }

    except Exception as e:
        return {"message": "Error during visualization.", "error": str(e)}

if __name__ == "__main__":
    import sys
    input_file = sys.argv[1]
    input_file1 = sys.argv[2]
    output_dir = sys.argv[3]

    result = visualize_model_benchmarking(input_file, input_file1, output_dir)
    print(json.dumps(result))
