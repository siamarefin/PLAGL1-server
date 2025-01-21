import pandas as pd
import numpy as np
import os

def z_score_normalize(df):
    """
    Normalize a DataFrame using Z-score normalization.
    """
    # Ensure the DataFrame contains only numeric data
    numeric_df = df.select_dtypes(include='number')

    # Calculate Z-score normalization
    normalized_df = (numeric_df - numeric_df.mean()) / numeric_df.std()

    return normalized_df


def process_file(input_file, output_dir):
    """
    Process the input file to normalize data and save results.
    """
    try:
        # Read the input CSV file
        main_df = pd.read_csv(input_file)
        
        # Check if 'condition' column exists
        if 'condition' not in main_df.columns:
            raise ValueError("The input file must contain a 'condition' column.")

        # Drop 'condition' column for normalization
        main_df_d = main_df.drop(columns=['condition'])
        
        # Normalize the data
        main_df_norm = z_score_normalize(main_df_d)
        
        # Reattach the 'condition' column
        main_df_norm['condition'] = main_df['condition']

        # Define output file paths
        normalized_file = os.path.join(output_dir, "z_score_normalized_data.csv")
        
        # Save normalized data
        main_df_norm.to_csv(normalized_file, index=False)
        
        return {
            "message": "Normalization completed successfully.",
            "normalized_file": normalized_file
        }
    except Exception as e:
        return {
            "message": "Error during normalization.",
            "error": str(e)
        }



import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import os
from datetime import datetime

# Set random seed for reproducibility
random_seed = 123

# Function for visualization using dimensionality reduction (PCA, t-SNE, UMAP)
def visualize_dimensionality_reduction(input_file, output_dir):
    try:
        # Read the input data
        df = pd.read_csv(input_file)

        # Check for 'condition' column
        if 'condition' not in df.columns:
            raise ValueError("The input file must contain a 'condition' column.")

        X = df.drop(columns=['condition'])  # Exclude the target variable
        y = df['condition']  # Target variable (condition)

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # --- PCA ---
        pca = PCA(n_components=2, random_state=random_seed)
        pca_result = pca.fit_transform(X)
        pca_df = pd.DataFrame(data=pca_result, columns=['PCA1', 'PCA2'])
        pca_df['condition'] = y.values

        # Plot PCA
        pca_png = os.path.join(output_dir, "PCA_plot.png")
        pca_pdf = os.path.join(output_dir, "PCA_plot.pdf")
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='PCA1', y='PCA2', hue='condition', data=pca_df, palette='viridis')
        plt.title('PCA of MPXV Data')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.grid()
        plt.legend(title='Condition')
        plt.savefig(pca_png)
        plt.savefig(pca_pdf)
        plt.close()

        # --- t-SNE ---
        def set_perplexity(n_samples):
            """Set appropriate perplexity based on the number of samples."""
            return min(30, max(5, n_samples // 3))

        # Get appropriate perplexity
        n_samples = X.shape[0]
        perplexity_value = set_perplexity(n_samples)

        tsne = TSNE(n_components=2, perplexity=perplexity_value, n_iter=300, random_state=random_seed)
        tsne_result = tsne.fit_transform(X)
        tsne_df = pd.DataFrame(data=tsne_result, columns=['TSNE1', 'TSNE2'])
        tsne_df['condition'] = y.values

        # Plot t-SNE
        tsne_png = os.path.join(output_dir, "tSNE_plot.png")
        tsne_pdf = os.path.join(output_dir, "tSNE_plot.pdf")
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='TSNE1', y='TSNE2', hue='condition', data=tsne_df, palette='viridis')
        plt.title('t-SNE of MPXV Data')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.grid()
        plt.legend(title='Condition')
        plt.savefig(tsne_png)
        plt.savefig(tsne_pdf)
        plt.close()

        # --- UMAP ---
        umap_model = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=random_seed)
        umap_result = umap_model.fit_transform(X)
        umap_df = pd.DataFrame(data=umap_result, columns=['UMAP1', 'UMAP2'])
        umap_df['condition'] = y.values

        # Plot UMAP
        umap_png = os.path.join(output_dir, "UMAP_plot.png")
        umap_pdf = os.path.join(output_dir, "UMAP_plot.pdf")
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='UMAP1', y='UMAP2', hue='condition', data=umap_df, palette='viridis')
        plt.title('UMAP of MPXV Data')
        plt.xlabel('UMAP Component 1')
        plt.ylabel('UMAP Component 2')
        plt.grid()
        plt.legend(title='Condition')
        plt.savefig(umap_png)
        plt.savefig(umap_pdf)
        plt.close()

        # --- Combined Plot ---
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Plot PCA
        sns.scatterplot(
            x='PCA1', y='PCA2', hue='condition', data=pca_df, palette='viridis', ax=axes[0]
        )
        axes[0].set_title('PCA of Data')
        axes[0].set_xlabel('Principal Component 1')
        axes[0].set_ylabel('Principal Component 2')
        axes[0].legend(title='Condition')

        # Plot t-SNE
        sns.scatterplot(
            x='TSNE1', y='TSNE2', hue='condition', data=tsne_df, palette='viridis', ax=axes[1]
        )
        axes[1].set_title('t-SNE of Data')
        axes[1].set_xlabel('t-SNE Component 1')
        axes[1].set_ylabel('t-SNE Component 2')
        axes[1].legend(title='Condition')

        # Plot UMAP
        sns.scatterplot(
            x='UMAP1', y='UMAP2', hue='condition', data=umap_df, palette='viridis', ax=axes[2]
        )
        axes[2].set_title('UMAP of Data')
        axes[2].set_xlabel('UMAP Component 1')
        axes[2].set_ylabel('UMAP Component 2')
        axes[2].legend(title='Condition')

        # Generate a unique timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Adjust layout
        plt.tight_layout()

        # Save the combined plots
        combined_png = os.path.join(output_dir, f"dimensionality_reduction_combined_{timestamp}.png")
        combined_pdf = os.path.join(output_dir, f"dimensionality_reduction_combined_{timestamp}.pdf")
        plt.savefig(combined_png)
        plt.savefig(combined_pdf)
        plt.close()

        return {
            "message": "Dimensionality reduction visualizations created successfully.",
            "PCA": {"png": pca_png, "pdf": pca_pdf},
            "tSNE": {"png": tsne_png, "pdf": tsne_pdf},
            "UMAP": {"png": umap_png, "pdf": umap_pdf},
            "Combined": {"png": combined_png, "pdf": combined_pdf}
        }

    except Exception as e:
        return {
            "message": "Error during visualization.",
            "error": str(e)
        }


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def plot_correlation_clustermap(input_file, output_dir, drop_column):
    try:
        # Read the input data
        df = pd.read_csv(input_file)

        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Drop the specified column
        df_cor = df.drop(columns=[drop_column])

        # Compute the Pearson correlation matrix
        correlation_matrix = df_cor.corr(method='pearson')

        # Save the highly correlated pairs to a CSV file
        corr_pairs = correlation_matrix.unstack().reset_index()
        corr_pairs.columns = ['Feature 1', 'Feature 2', 'Correlation']
        corr_pairs = corr_pairs[
            (corr_pairs['Feature 1'] != corr_pairs['Feature 2']) & 
            (corr_pairs['Feature 1'] < corr_pairs['Feature 2'])
        ]
        corr_pairs = corr_pairs.sort_values(by='Correlation', ascending=False)
        corr_csv_path = os.path.join(output_dir, 'Highly_Correlated_Features.csv')
        corr_pairs.to_csv(corr_csv_path, index=False)

        # Create a clustermap
        clustermap = sns.clustermap(
            correlation_matrix,
            annot=False,
            cmap='coolwarm',
            vmin=-1,
            vmax=1,
            cbar_kws={"shrink": .8},
            method='average'
        )

        # Set title
        # plt.suptitle('Pearson Correlation Clustermap', fontsize=16)

        # Save the plot as a PDF and PNG file
        pdf_path = os.path.join(output_dir, 'Pearson_Correlation_Clustermap.pdf')
        png_path = os.path.join(output_dir, 'Pearson_Correlation_Clustermap.png')
        clustermap.savefig(pdf_path)
        clustermap.savefig(png_path)

        # Close the plot
        plt.close()

        return {
            "message": "Correlation clustermap created successfully.",
            "output_files": {
                "correlation_csv": corr_csv_path,
                "correlation_pdf": pdf_path,
                "correlation_png": png_path
            }
        }

    except Exception as e:
        return {
            "message": "Error generating correlation clustermap.",
            "error": str(e)
        }


import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
import os
import json

def feature_selection_and_model(input_file, output_dir, feature_ratio):
    try:
        # Load data
        df = pd.read_csv(input_file)

        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Split data
        X = df.drop(columns=['condition'])
        y = df['condition']

        # Apply RFE with RF
        rf_model = RandomForestClassifier(random_state=123)
        num_features_to_select = int(X.shape[1] * feature_ratio)
        rfe = RFE(estimator=rf_model, n_features_to_select=num_features_to_select, step=1)
        rfe.fit(X, y)

        # Select features
        selected_features = X.columns[rfe.support_]
        X_selected = X[selected_features].copy()  # Reduced dataset with selected features
        X_selected['condition'] = y  # Add condition column back

        # Save selected features list
        selected_features_path = os.path.join(output_dir, "selected_features.csv")
        X_selected.to_csv(selected_features_path, index=False)

        # Train and evaluate
        rf_model_reduced = RandomForestClassifier(random_state=123)
        cv_scores = cross_val_score(rf_model_reduced, X_selected[selected_features], y, cv=5, scoring='roc_auc')

        # Return results
        result = {
            "message": "Feature selection and model training completed successfully.",
            "output_files": {
                "selected_features_csv": selected_features_path
            },
            "model_metrics": {
                "cross_validation_auc": cv_scores.mean()
            }
        }
        return json.dumps(result)

    except Exception as e:
        return json.dumps({
            "message": "Error during feature selection and model training.",
            "error": str(e)
        })
    




    

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

# Declare global variable for best models
best_models = {}

# Initialize storage for metrics and results
metrics = []
roc_curves = {}
pr_curves = {}

def benchmark_models(input_file,output_dir):
    global best_models  # Declare best_models as global to store results across API calls
    

    try:
        # Load dataset
        df = pd.read_csv(input_file)
        X = df.drop(columns=['condition'])
        y = df['condition']

        # Initialize Stratified Cross-Validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)

        

        for name, model in classifiers.items():
            print(f"Tuning {name}...")

            # Cross-validation predictions for original model
            y_pred_proba_original = cross_val_predict(
                model, X, y, cv=cv, method='predict_proba', n_jobs=-1
            )[:, 1]
            original_auc = roc_auc_score(y, y_pred_proba_original)
            original_auprc = average_precision_score(y, y_pred_proba_original)

            # Calculate ROC and Precision-Recall curves for the original model
            fpr, tpr, _ = roc_curve(y, y_pred_proba_original)
            precision, recall, _ = precision_recall_curve(y, y_pred_proba_original)
            roc_curves[name] = (fpr, tpr)
            pr_curves[name] = (precision, recall)

            # Optimize threshold for F1-score
            thresholds = np.arange(0, 1, 0.01)
            best_f1, best_threshold = max(
                (f1_score(y, (y_pred_proba_original > t).astype(int)), t) for t in thresholds
            )
            y_pred = (y_pred_proba_original > best_threshold).astype(int)
            accuracy = accuracy_score(y, y_pred)
            precision_at_threshold = precision_score(y, y_pred)
            recall_at_threshold = recall_score(y, y_pred)

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
                    best_auc = tuned_auc
                    best_auprc = tuned_auprc
                    fpr, tpr, _ = roc_curve(y, y_pred_proba_tuned)
                    precision, recall, _ = precision_recall_curve(y, y_pred_proba_tuned)
                    roc_curves[name] = (fpr, tpr)
                    pr_curves[name] = (precision, recall)
                    print(f"{name}: Tuned model performed better (AUROC: {tuned_auc:.4f}, AUPRC: {tuned_auprc:.4f})")
                else:
                    best_model = model
                    best_auc = original_auc
                    best_auprc = original_auprc
                    print(f"{name}: Original model retained (AUROC: {original_auc:.4f}, AUPRC: {original_auprc:.4f})")
            else:
                best_model = model
                best_auc = original_auc
                best_auprc = original_auprc
                print(f"{name}: No hyperparameter tuning. Original model AUROC: {original_auc:.4f}, AUPRC: {original_auprc:.4f}")

            # Store the best model and metrics
            best_models[name] = best_model

            # Store metrics
            metrics.append({
                'Model': name,
                'AUPRC': best_auprc,
                'AUROC': best_auc,
                'Precision': precision_at_threshold,
                'Recall': recall_at_threshold,
                'F1-Score': best_f1,
                'Accuracy': accuracy,
            })
        # Convert metrics to a DataFrame
        metrics_df = pd.DataFrame(metrics)
        metrics_df = metrics_df.sort_values(by=['AUPRC'], ascending=False).reset_index(drop=True)
         # Save metrics as CSV
        metrics_path = os.path.join(output_dir, "model_benchmarking_results.csv")
        metrics_df.to_csv(metrics_path, index=False)
       
        # Debug: Log what is being returned
        print({
            "metrics": metrics_df.to_dict(orient="records"),
            "metrics_path": metrics_path
        })



        print(" ")

        print(best_models)


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
            "metrics": metrics_df.to_dict(orient="records"),
            "metrics_path": metrics_path
        }

    except Exception as e:
        return f"Error: {str(e)}"




import pandas as pd
import matplotlib.pyplot as plt

def get_model_and_importance_with_top10(metrics_df, best_models, reduced_df, selected_model_name, output_dir, feature_names=None):
    """
    Analyze feature importance for a selected model and extract top 10 features.
    """
    # Ensure the selected model exists
    if selected_model_name not in best_models:
        raise ValueError(f"Model '{selected_model_name}' not found in best_models.")

    # Retrieve the selected model
    selected_model = best_models[selected_model_name]
    print(f"\nAnalyzing feature importance for model: {selected_model_name}")

    # Handle feature names
    if feature_names is None:
        feature_names = reduced_df.drop(columns=['condition']).columns.tolist()

    # Compute feature importance
    if hasattr(selected_model, "feature_importances_"):
        # Tree-based models
        importance_scores = selected_model.feature_importances_
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance_scores
        }).sort_values(by='Importance', ascending=False)

    elif hasattr(selected_model, "coef_"):
        # Linear models
        coef_scores = selected_model.coef_.flatten()
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': coef_scores
        }).sort_values(by='Importance', ascending=False)

    else:
        raise ValueError(f"Model '{selected_model_name}' does not support feature importance or coefficients.")

    # Select top 10 features
    top10_features = importance_df.head(10)
    print("\nTop 10 Feature Importance Scores:")
    print(top10_features)

    # Plot top 10 feature importance
    plt.figure(figsize=(8, 5))
    plt.barh(top10_features['Feature'], top10_features['Importance'], color='skyblue', align='center')
    plt.gca().invert_yaxis()
    plt.title(f"Top 10 Feature Importance for {selected_model_name}")
    plt.xlabel('Importance Score')
    plt.ylabel('Feature')
    plt.tight_layout()

    # Save the plot
    plot_path = f"{output_dir}/top10_feature_importance_{selected_model_name}.png"
    plt.savefig(plot_path)
    plt.close()

    # Extract the top 10 features from the reduced DataFrame
    top10_feature_names = top10_features['Feature'].tolist()

    # Include the 'condition' column along with the top 10 features
    columns_to_include = top10_feature_names + ['condition']
    top10_df = reduced_df[columns_to_include]

    # Save the top 10 DataFrame
    top10_path = f"{output_dir}/top10_features_{selected_model_name}.csv"
    top10_df.to_csv(top10_path, index=False)

    return {
        "top10_features_path": top10_path,
        "top10_plot_path": plot_path,
        "top10_features": top10_features.to_dict(orient="records")
    }




