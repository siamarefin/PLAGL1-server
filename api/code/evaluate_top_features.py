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


def plot_performance_metrics(roc_curves, pr_curves, output_dir):
    try:
        # Sort ROC and Precision-Recall curves by AUC in descending order
        roc_curves.sort(key=lambda x: x[2], reverse=True)
        pr_curves.sort(key=lambda x: x[2], reverse=True)

        # Create a figure with 2 subplots in landscape orientation
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # --- Precision-Recall Curves (AUPRC) ---
        for precision, recall, pr_auc, n_features in pr_curves:
            axes[0].plot(recall, precision, lw=1.75, label=f'{n_features} Genes Model (AUPRC = {pr_auc:.2f})')
        axes[0].set_title('Precision-Recall Curves with Varying Number of Features')
        axes[0].set_xlabel('Recall')
        axes[0].set_ylabel('Precision')
        axes[0].set_xlim([0.0, 1.0])
        axes[0].set_ylim([0.0, 1.05])
        axes[0].legend(loc='lower left', fontsize=9, frameon=False)

        # --- ROC Curves (AUROC) ---
        for fpr, tpr, roc_auc, n_features in roc_curves:
            axes[1].plot(fpr, tpr, lw=1.75, label=f'{n_features} Genes Model (AUROC = {roc_auc:.2f})')
        axes[1].plot([0, 1], [0, 1], 'k--', label='Random Chance')  # Add random chance line
        axes[1].set_title('ROC Curves with Varying Number of Features')
        axes[1].set_xlabel('False Positive Rate')
        axes[1].set_ylabel('True Positive Rate')
        axes[1].set_xlim([0.0, 1.0])
        axes[1].set_ylim([0.0, 1.05])
        axes[1].legend(loc='lower right', fontsize=9, frameon=False)

        # Add a main title for the combined figure
        fig.suptitle('Performance Metrics with Varying Number of Features', fontsize=16, y=1.02)

        # Adjust layout for landscape view
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        # Save the combined figure
        png_path = os.path.join(output_dir, "performance_metrics_landscape_top.png")
        pdf_path = os.path.join(output_dir, "performance_metrics_landscape_top.pdf")
        plt.savefig(png_path, dpi=300, bbox_inches="tight")
        plt.savefig(pdf_path, dpi=300, bbox_inches="tight")
        plt.close()

        print("Plots saved to:", png_path, "and", pdf_path)

    except Exception as e:
        print("Error generating plot:", str(e))
    except Exception as e:
        print("Error generating plot:", str(e))



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

        print("Performance metrics saved to:", metrics_csv_path)

        # Identify the best-performing model based on a selected metric (e.g., Accuracy or AUPRC)
        best_row = metrics_df.loc[metrics_df['Accuracy'].idxmax()]
        best_n_features = int(best_row['Number of Features'])

        # Select top features for the best model
        best_features = list(top10_df_array[:best_n_features])
        final_df = top10_df[best_features + ['condition']]

        # Save the final selected features to a CSV file
        final_features_csv_path = os.path.join(output_dir, f"{model_name}_best_features.csv")
        final_df.to_csv(final_features_csv_path, index=False)

        print(f"Best model with {best_n_features} features saved to: {final_features_csv_path}")

        
        # Plot performance metrics
        plot_performance_metrics(roc_curves, pr_curves, output_dir)

        return {
            "message": "Feature evaluation completed successfully.",
            "metrics_file": metrics_csv_path,
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
