import os
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_predict
from sklearn.metrics import (
    roc_auc_score, average_precision_score, roc_curve, precision_recall_curve,
    accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from joblib import dump

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

def evaluate_model_performance(input_file, output_dir, model_name):
    try:
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Load data
        final_df = pd.read_csv(input_file)

        # Prepare data
        X = final_df.drop(columns=['condition'])
        y = final_df['condition']

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=123
        )

        # Define cross-validation and model
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)
        model = classifiers[model_name]
        param_grid = param_grids.get(model_name, {})

        # Hyperparameter tuning
        grid_search = GridSearchCV(model, param_grid, cv=cv, scoring='roc_auc', n_jobs=-1)
        grid_search.fit(X_train, y_train)

        # Use the best model
        tuned_model = grid_search.best_estimator_

        # Evaluate on training data
        y_pred_proba_train_cv = cross_val_predict(
            tuned_model, X_train, y_train, cv=cv, method='predict_proba', n_jobs=-1
        )[:, 1]
        train_roc_auc_cv = roc_auc_score(y_train, y_pred_proba_train_cv)
        train_pr_auc_cv = average_precision_score(y_train, y_pred_proba_train_cv)

        # Train on the full training set and evaluate on test set
        tuned_model.fit(X_train, y_train)
        y_pred_proba_test = tuned_model.predict_proba(X_test)[:, 1]
        y_pred_test = (y_pred_proba_test > 0.5).astype(int)

        # Calculate metrics on the test set
        test_roc_auc = roc_auc_score(y_test, y_pred_proba_test)
        test_pr_auc = average_precision_score(y_test, y_pred_proba_test)
        test_accuracy = accuracy_score(y_test, y_pred_test)
        test_f1 = f1_score(y_test, y_pred_test)
        test_precision = precision_score(y_test, y_pred_test)
        test_recall = recall_score(y_test, y_pred_test)

        # Save metrics to a text file
        metrics_path = os.path.join(output_dir, f"{model_name}_evaluation_metrics.txt")
        with open(metrics_path, 'w') as f:
            f.write(f"Train AUROC: {train_roc_auc_cv:.4f}\n")
            f.write(f"Train AUPRC: {train_pr_auc_cv:.4f}\n")
            f.write(f"Test AUROC: {test_roc_auc:.4f}\n")
            f.write(f"Test AUPRC: {test_pr_auc:.4f}\n")
            f.write(f"Test Accuracy: {test_accuracy:.4f}\n")
            f.write(f"Test F1-Score: {test_f1:.4f}\n")
            f.write(f"Test Precision: {test_precision:.4f}\n")
            f.write(f"Test Recall: {test_recall:.4f}\n")

        # Visualize performance metrics
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba_test)
        axes[0].plot(recall, precision, label=f'PR Curve (AUPRC = {test_pr_auc:.2f})')
        axes[0].set_title('Precision-Recall Curve')
        axes[0].set_xlabel('Recall')
        axes[0].set_ylabel('Precision')
        axes[0].legend(loc='lower left')

        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba_test)
        axes[1].plot(fpr, tpr, label=f'ROC Curve (AUROC = {test_roc_auc:.2f})')
        axes[1].plot([0, 1], [0, 1], 'k--', label='Random Chance')
        axes[1].set_title('Receiver Operating Characteristic (ROC) Curve')
        axes[1].set_xlabel('False Positive Rate')
        axes[1].set_ylabel('True Positive Rate')
        axes[1].legend(loc='lower right')

        fig.suptitle('Performance of the Final Model', fontsize=16, y=1)
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        # Save the plot
        plot_path = os.path.join(output_dir, f"{model_name}_performance_plot.png")
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        # Save confusion matrix
        cm = confusion_matrix(y_test, y_pred_test)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative', 'Positive'])
        disp.plot(cmap='Blues', values_format='d')
        cm_path = os.path.join(output_dir, f"{model_name}_confusion_matrix.png")
        plt.savefig(cm_path, dpi=300, bbox_inches="tight")
        plt.close()

        # Save the final model
        model_path = os.path.join(output_dir, f"{model_name}_final_model.joblib")
        dump(tuned_model, model_path)

        print(f"Metrics saved to: {metrics_path}")
        print(f"Performance plot saved to: {plot_path}")
        print(f"Confusion matrix saved to: {cm_path}")
        print(f"Model saved to: {model_path}")

        return {
            "message": "Model evaluation completed successfully.",
            "metrics_file": metrics_path,
            "plot_file": plot_path,
            "confusion_matrix_file": cm_path,
            "model_file": model_path
        }

    except Exception as e:
        return {"message": "Error during model evaluation.", "error": str(e)}

if __name__ == "__main__":
    import sys
    input_file = sys.argv[1]
    output_dir = sys.argv[2]
    model_name = sys.argv[3]
    result = evaluate_model_performance(input_file, output_dir, model_name)
    print(result)
