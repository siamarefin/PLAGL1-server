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



import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import  umap 
import os

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


        # Adjust layout
        plt.tight_layout()

        # Save the combined plots
        combined_png = os.path.join(output_dir, f"dimensionality_reduction_combined.png")
        combined_pdf = os.path.join(output_dir, f"dimensionality_reduction_combined.png")
        plt.savefig(combined_png)
        plt.savefig(combined_pdf)
        plt.close()

        return {
            "message": "Dimensionality reduction visualizations created successfully.",
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



import pandas as pd
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
        combined_png = os.path.join(output_dir, f"visualize_dimensions_10_feature_{timestamp}.png")
        combined_pdf = os.path.join(output_dir, f"visualize_dimensions_10_feature_{timestamp}.pdf")
        plt.savefig(combined_png)
        plt.savefig(combined_pdf)
        plt.close()

        return {
            "message": "Dimensionality reduction visualizations created successfully.",
            "Combined": {"png": combined_png, "pdf": combined_pdf}
        }

    except Exception as e:
        return {
            "message": "Error during visualization.",
            "error": str(e)
        }



import pandas as pd
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, average_precision_score,
    f1_score, accuracy_score, roc_curve, precision_score, recall_score
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import matplotlib.pyplot as plt
import os

def rank_features(input_file, selected_model_name, param_grids, classifiers, output_dir):
    """
    Function to rank features based on individual model performance metrics and plot performance metrics.
    """
    try:
        # Read the input data
        df = pd.read_csv(input_file)

        # Ensure 'condition' column exists
        if 'condition' not in df.columns:
            raise ValueError("The input file must contain a 'condition' column.")

        # Prepare data
        X = df.drop(columns=['condition'])  # Feature data
        y = df['condition']  # Target variable
        top10_features = X.columns.tolist()

        # Initialize Stratified Cross-Validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)

        # Hyperparameter grid for the selected model
        if selected_model_name not in param_grids:
            raise ValueError(f"Parameter grid not found for the model: {selected_model_name}")
        param_grid = param_grids[selected_model_name]

        if selected_model_name not in classifiers:
            raise ValueError(f"Classifier not found for the model: {selected_model_name}")
        model = classifiers[selected_model_name]

        # Store results for metrics
        metrics_scores = []
        roc_auc_scores = {}
        predictions = {}
        auprc_scores = {}

        # Rank features individually
        for feature in top10_features:
            print(f"Processing feature: {feature}")

            # Prepare data for the single feature
            X_single = X[[feature]]  # Select only the current feature

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

            # Store Precision-Recall and ROC metrics
            roc_auc_scores[feature] = roc_auc
            predictions[feature] = y_pred_proba
            auprc_scores[feature] = auprc

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

        # Convert metrics to a DataFrame
        metrics_df = pd.DataFrame(metrics_scores)

        # Sort features based on the primary metric in descending order
        metrics_df.sort_values(by='AUPRC', ascending=False, inplace=True)

        # Save the sorted results to a CSV file
        output_file = os.path.join(output_dir, f'feature_metrics_ranking.csv')
        metrics_df.to_csv(output_file, index=False)

        # Plot Precision-Recall Curves (AUPRC) and ROC Curves (AUROC)
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))  # Adjust width and height for landscape view

        # --- Precision-Recall Curves (AUPRC) ---
        # Sort features by AUPRC score in descending order
        sorted_auprc_features = sorted(auprc_scores.items(), key=lambda x: x[1], reverse=True)

        for feature, auprc in sorted_auprc_features:
            y_pred_proba_cv = predictions[feature]
            precision, recall, _ = precision_recall_curve(y, y_pred_proba_cv)
            axes[0].plot(recall, precision, lw=1.75, label=f'{feature} (AUPRC = {auprc:.2f})')

        axes[0].set_title('Precision-Recall Curves for Individual Features')
        axes[0].set_xlabel('Recall')
        axes[0].set_ylabel('Precision')
        axes[0].set_xlim([0.0, 1.0])
        axes[0].set_ylim([0.0, 1.05])
        axes[0].legend(loc='lower left', fontsize=8, frameon=False)

        # --- ROC Curves (AUROC) ---
        # Sort features by ROC AUC score in descending order
        sorted_features = sorted(roc_auc_scores.items(), key=lambda x: x[1], reverse=True)

        for feature, auc_score in sorted_features:
            fpr, tpr, _ = roc_curve(y, predictions[feature])
            axes[1].plot(fpr, tpr, lw=1.75, label=f'{feature} (AUROC = {auc_score:.2f})')

        # Plot diagonal line for random chance in ROC subplot
        axes[1].plot([0, 1], [0, 1], 'k--', label='Random Chance')
        axes[1].set_title('ROC Curves for Individual Features')
        axes[1].set_xlabel('False Positive Rate')
        axes[1].set_ylabel('True Positive Rate')
        axes[1].set_xlim([0.0, 1.0])
        axes[1].set_ylim([0.0, 1.05])
        axes[1].legend(loc='lower right', fontsize=8, frameon=False)

        # Add a main title for the figure
        fig.suptitle('Performance Metrics for Individual Features', fontsize=16, y=1)

        # Adjust layout for landscape view
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for spacing and main title

        # Save the figure
        plot_png_path = os.path.join(output_dir, f'feature_performance_landscape_{selected_model_name}.png')
        plot_pdf_path = os.path.join(output_dir, f'feature_performance_landscape_{selected_model_name}.pdf')
        plt.savefig(plot_png_path, dpi=300, bbox_inches='tight')
        plt.savefig(plot_pdf_path)
        plt.close()

        return {
            "message": "Feature ranking and plotting completed successfully.",
            "ranking_file": output_file,
            "plot_png": plot_png_path,
            "plot_pdf": plot_pdf_path,
            "metrics": metrics_df.to_dict(orient="records")
        }

    except Exception as e:
        return {"message": "Error during feature ranking and plotting.", "error": str(e)}



import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_predict
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score, f1_score,
    roc_curve, precision_recall_curve, precision_score, recall_score
)
import matplotlib.pyplot as plt
import os

def evaluate_model_with_features(input_file, selected_model, param_grids, classifiers, output_dir):
    """
    Function to evaluate models with varying numbers of top features and plot performance metrics.
    """
    try:
        # Load the dataset
        df = pd.read_csv(input_file)

        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Validate the input
        if 'condition' not in df.columns:
            raise ValueError("The input file must contain a 'condition' column.")
        
        # Prepare data
        X_full = df.drop(columns=['condition'])
        y = df['condition']
        top_features = X_full.columns.tolist()

        # Initialize Stratified Cross-Validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)

        # Initialize storage for curves and performance metrics
        roc_curves = []
        pr_curves = []
        performance_metrics = []

        # Loop over the number of features from 10 to 1
        for n_features in range(10, 0, -1):
            print(f"Evaluating model with top {n_features} features...")

            # Create a subset of features
            selected_features = top_features[:n_features]
            X = X_full[selected_features]

            # Get the model and hyperparameters
            if selected_model not in classifiers or selected_model not in param_grids:
                raise ValueError(f"Invalid model: {selected_model}")
            model = classifiers[selected_model]
            param_grid = param_grids[selected_model]

            # Perform GridSearchCV for hyperparameter tuning
            grid_search = GridSearchCV(model, param_grid, cv=cv, scoring='roc_auc', n_jobs=-1)
            grid_search.fit(X, y)
            final_model = grid_search.best_estimator_

            # Cross-validation predictions
            y_pred_proba = cross_val_predict(final_model, X, y, cv=cv, method='predict_proba')[:, 1]
            y_pred = (y_pred_proba > 0.5).astype(int)

            # Calculate metrics
            roc_auc = roc_auc_score(y, y_pred_proba)
            pr_auc = average_precision_score(y, y_pred_proba)
            f1 = f1_score(y, y_pred)
            accuracy = accuracy_score(y, y_pred)
            precision = precision_score(y, y_pred)
            recall = recall_score(y, y_pred)

            # Store ROC and Precision-Recall curves
            fpr, tpr, _ = roc_curve(y, y_pred_proba)
            precision_curve, recall_curve, _ = precision_recall_curve(y, y_pred_proba)
            roc_curves.append((fpr, tpr, roc_auc, n_features))
            pr_curves.append((precision_curve, recall_curve, pr_auc, n_features))

            # Store performance metrics
            performance_metrics.append({
                'Number of Features': n_features,
                'AUPRC': pr_auc,
                'AUROC': roc_auc,
                'F1 Score': f1,
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall
            })

        # Save performance metrics to a CSV file
        metrics_df = pd.DataFrame(performance_metrics)
        metrics_df.sort_values(by='AUPRC', ascending=False, inplace=True)
        metrics_csv = os.path.join(output_dir, 'gene_subset_model_performance_cv.csv')
        metrics_df.to_csv(metrics_csv, index=False)
         # Ensure that Precision-Recall curves are computed
        if not pr_curves:
            raise ValueError("pr_curves is empty. Ensure the Precision-Recall curves are computed before selecting features.")

        # Select the top model based on AUPRC
        best_pr_curve = max(pr_curves, key=lambda x: x[2])  # Get the curve with the highest AUPRC
        best_pr_n_features = best_pr_curve[3]  # Number of features for the best AUPRC
        print(f"Best model based on AUPRC uses {best_pr_n_features} features.")

        # Ensure selected features are valid
        selected_features = top_features[:best_pr_n_features]
        missing_features = [feature for feature in selected_features if feature not in df.columns]
        if missing_features:
            raise ValueError(f"The following features are missing in the DataFrame: {missing_features}")

        # Create the final DataFrame with selected features
        final_df = df[selected_features + ['condition']]
        final_df_path = os.path.join(output_dir, 'final_selected_features_auprc.csv')
        final_df.to_csv(final_df_path, index=False)

        # --- Plot Precision-Recall and ROC Curves ---
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Sort ROC curves by AUC
        roc_curves.sort(key=lambda x: x[2], reverse=True)
        # Sort PR curves by PR AUC
        pr_curves.sort(key=lambda x: x[2], reverse=True)

        # Plot Precision-Recall Curves
        for precision, recall, pr_auc, n_features in pr_curves:
            axes[0].plot(recall, precision, lw=1.75, label=f'{n_features} Features (AUPRC = {pr_auc:.2f})')
        axes[0].set_title('Precision-Recall Curves')
        axes[0].set_xlabel('Recall')
        axes[0].set_ylabel('Precision')
        axes[0].legend(loc='lower left', fontsize=9, frameon=False)

        # Plot ROC Curves
        for fpr, tpr, roc_auc, n_features in roc_curves:
            axes[1].plot(fpr, tpr, lw=1.75, label=f'{n_features} Features (AUROC = {roc_auc:.2f})')
        axes[1].plot([0, 1], [0, 1], 'k--', label='Random Chance')
        axes[1].set_title('ROC Curves')
        axes[1].set_xlabel('False Positive Rate')
        axes[1].set_ylabel('True Positive Rate')
        axes[1].legend(loc='lower right', fontsize=9, frameon=False)

        # Add a main title for the figure
        fig.suptitle('Performance Metrics with Varying Number of Features', fontsize=16, y=1)
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        # Save the plots
        plot_png = os.path.join(output_dir, 'performance_metrics_landscape.png')
        plot_pdf = os.path.join(output_dir, 'performance_metrics_landscape.pdf')
        plt.savefig(plot_png, dpi=300, bbox_inches='tight')
        plt.savefig(plot_pdf)
        plt.close()

        return {
            "message": "Feature evaluation completed successfully.",
            "metrics_file": metrics_csv,
            "plot_png": plot_png,
            "plot_pdf": plot_pdf,
            "metrics": metrics_df.to_dict(orient="records")
        }

    except Exception as e:
        return {"message": "Error during feature evaluation.", "error": str(e)}


# Function for visualization using dimensionality reduction (PCA, t-SNE, UMAP)
def visualize_dimensionality_reduction_final(input_file, output_dir):
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
        pca_png = os.path.join(output_dir, "PCA_plot_final.png")
        pca_pdf = os.path.join(output_dir, "PCA_plot_final.pdf")
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
        tsne_png = os.path.join(output_dir, "tSNE_plot_final.png")
        tsne_pdf = os.path.join(output_dir, "tSNE_plot_final.pdf")
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
        umap_png = os.path.join(output_dir, "UMAP_plot_final.png")
        umap_pdf = os.path.join(output_dir, "UMAP_plot_final.pdf")
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
        combined_png = os.path.join(output_dir, f"visualize_dimensions_final_{timestamp}.png")
        combined_pdf = os.path.join(output_dir, f"visualize_dimensions_final_{timestamp}.pdf")
        plt.savefig(combined_png)
        plt.savefig(combined_pdf)
        plt.close()

        return {
            "message": "Dimensionality reduction visualizations created successfully.",
            "Combined": {"png": combined_png, "pdf": combined_pdf}
        }

    except Exception as e:
        return {
            "message": "Error during visualization.",
            "error": str(e)
        }
    



from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_predict
from sklearn.metrics import (
    roc_auc_score, average_precision_score, roc_curve, precision_recall_curve,
    accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
)
from joblib import dump
import matplotlib.pyplot as plt
import os

def evaluate_final_model(final_df_path, selected_model, param_grids, classifiers, output_dir):
    """
    Function to train, validate, and test the final model, and save results and plots.
    """
    try:
        # Load the final dataset
        final_df = pd.read_csv(final_df_path)

        # Ensure 'condition' column exists
        if 'condition' not in final_df.columns:
            raise ValueError("The input file must contain a 'condition' column.")

        # Prepare data
        X = final_df.drop(columns=['condition'])
        y = final_df['condition']

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=123
        )
        print(f"Training set size: {X_train.shape}, Testing set size: {X_test.shape}")

        # Define cross-validation and the model
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)
        default_model = classifiers[selected_model]
        param_grid = param_grids[selected_model]

        # Hyperparameter tuning with GridSearchCV
        print(f"Tuning hyperparameters for {selected_model}...")
        grid_search = GridSearchCV(
            default_model, param_grid, cv=cv, scoring='roc_auc', n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        tuned_model = grid_search.best_estimator_
        print(f"Best parameters for {selected_model}: {grid_search.best_params_}")

        # Cross-validation on the training set
        y_pred_proba_train_cv = cross_val_predict(
            tuned_model, X_train, y_train, cv=cv, method='predict_proba', n_jobs=-1
        )[:, 1]
        y_pred_train_cv = cross_val_predict(
            tuned_model, X_train, y_train, cv=cv, method='predict', n_jobs=-1
        )

        # Training set metrics
        train_roc_auc_cv = roc_auc_score(y_train, y_pred_proba_train_cv)
        train_pr_auc_cv = average_precision_score(y_train, y_pred_proba_train_cv)
        train_accuracy_cv = accuracy_score(y_train, y_pred_train_cv)
        train_f1_cv = f1_score(y_train, y_pred_train_cv)
        train_precision_cv = precision_score(y_train, y_pred_train_cv)
        train_recall_cv = recall_score(y_train, y_pred_train_cv)

        print(f"\nTraining Set Cross-Validation Metrics:")
        print(f"AUROC: {train_roc_auc_cv:.4f}, AUPRC: {train_pr_auc_cv:.4f}, Accuracy: {train_accuracy_cv:.4f}")

        # Train on the full training set
        tuned_model.fit(X_train, y_train)

        # Test set predictions and metrics
        y_pred_proba_test = tuned_model.predict_proba(X_test)[:, 1]
        y_pred_test = (y_pred_proba_test > 0.5).astype(int)
        test_roc_auc = roc_auc_score(y_test, y_pred_proba_test)
        test_pr_auc = average_precision_score(y_test, y_pred_proba_test)
        test_accuracy = accuracy_score(y_test, y_pred_test)
        test_f1 = f1_score(y_test, y_pred_test)
        test_precision = precision_score(y_test, y_pred_test)
        test_recall = recall_score(y_test, y_pred_test)

        print(f"\nTest Set Metrics:")
        print(f"AUROC: {test_roc_auc:.4f}, AUPRC: {test_pr_auc:.4f}, Accuracy: {test_accuracy:.4f}")

        # Save the model
        model_filename = os.path.join(output_dir, "final_model.joblib")
        dump(tuned_model, model_filename)
        print(f"Model saved as {model_filename}")

        # Plot Precision-Recall and ROC Curves
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba_test)
        axes[0].plot(recall, precision, label=f'PR Curve (AUPRC = {test_pr_auc:.2f})')
        axes[0].set_title('Precision-Recall Curve')
        axes[0].set_xlabel('Recall')
        axes[0].set_ylabel('Precision')
        axes[0].legend(loc='lower left')
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba_test)
        axes[1].plot(fpr, tpr, label=f'ROC Curve (AUROC = {test_roc_auc:.2f})')
        axes[1].plot([0, 1], [0, 1], 'k--', label='Random Chance')
        axes[1].set_title('ROC Curve')
        axes[1].set_xlabel('False Positive Rate')
        axes[1].set_ylabel('True Positive Rate')
        axes[1].legend(loc='lower right')
        fig.suptitle('Performance of the Final Model', fontsize=16, y=1)
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        # Save the plots
        pr_roc_png = os.path.join(output_dir, 'final_model_pr_roc_curves.png')
        plt.savefig(pr_roc_png, dpi=300, bbox_inches='tight')
        plt.close()

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred_test)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative', 'Positive'])
        cm_png = os.path.join(output_dir, 'final_model_confusion_matrix.png')
        disp.plot(cmap='Blues', values_format='d')
        plt.title("Confusion Matrix")
        plt.savefig(cm_png, dpi=300, bbox_inches='tight')
        plt.close()

        return {
            "message": "Final model evaluation completed successfully.",
            "train_metrics": {
                "AUROC": train_roc_auc_cv,
                "AUPRC": train_pr_auc_cv,
                "Accuracy": train_accuracy_cv,
                "F1-Score": train_f1_cv,
                "Precision": train_precision_cv,
                "Recall": train_recall_cv
            },
            "test_metrics": {
                "AUROC": test_roc_auc,
                "AUPRC": test_pr_auc,
                "Accuracy": test_accuracy,
                "F1-Score": test_f1,
                "Precision": test_precision,
                "Recall": test_recall
            },
            "model_path": model_filename,
            "pr_roc_plot": pr_roc_png,
            "confusion_matrix_plot": cm_png
        }

    except Exception as e:
        return {"message": "Error during final model evaluation.", "error": str(e)}
