import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

# Define classifiers
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

# Define parameter grids (use default values from grids)
default_params = {
    'Logistic Regression': {'C': 1.0, 'solver': 'liblinear'},
    'Extra Trees': {'n_estimators': 100, 'max_depth': None, 'min_samples_split': 2},
    'Random Forest': {'n_estimators': 100, 'max_depth': None, 'min_samples_split': 2},
    'XGBoost': {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 3},
    'Gradient Boosting': {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 3},
    'K Neighbors': {'n_neighbors': 5, 'weights': 'uniform'},
    'AdaBoost': {'n_estimators': 50, 'learning_rate': 1.0}
}


def get_model_and_importance_with_top10(model_name, input_file, output_dir):
    try:
        # Load selected features
        reduced_df = pd.read_csv(input_file)

        # Ensure the selected model exists
        if model_name not in classifiers:
            raise ValueError(f"Model '{model_name}' not found in classifiers.")

        # Retrieve the selected model and apply default parameters
        model = classifiers[model_name]
        if model_name in default_params:
            model.set_params(**default_params[model_name])

        # Handle feature names
        feature_names = reduced_df.drop(columns=['condition']).columns.tolist()

        # Train the model
        X = reduced_df.drop(columns=['condition'])
        y = reduced_df['condition']
        model.fit(X, y)

        # Compute feature importance
        if hasattr(model, "feature_importances_"):
            # Tree-based models
            importance_scores = model.feature_importances_
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importance_scores
            }).sort_values(by='Importance', ascending=False)

        elif hasattr(model, "coef_"):
            # Linear models
            coef_scores = model.coef_.flatten()
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': coef_scores
            }).sort_values(by='Importance', ascending=False)

        else:
            raise ValueError(f"Model '{model_name}' does not support feature importance computation.")

        # Select top 10 features
        top10_features = importance_df.head(10)

        # Save top 10 feature names to a text file
        top10_txt_path = os.path.join(output_dir, f"{model_name}_top10_features.txt")
        with open(top10_txt_path, "w") as f:
            f.write("Top 10 Feature Importance Scores:\n")
            f.write(top10_features.to_string(index=False))

        # Plot top 10 feature importance
        plt.figure(figsize=(8, 5))
        plt.barh(top10_features['Feature'], top10_features['Importance'], color='skyblue', align='center')
        plt.gca().invert_yaxis()
        plt.title(f"Top 10 Feature Importance for {model_name}")
        plt.xlabel('Importance Score')
        plt.ylabel('Feature')
        plt.tight_layout()

        # Save the plot
        plot_path = os.path.join(output_dir, f"{model_name}_top10_features.png")
        plt.savefig(plot_path)
        plt.close()

        # Save top 10 features DataFrame to a CSV file
        top10_csv_path = os.path.join(output_dir, f"{model_name}_top10_features.csv")
        top10_features.to_csv(top10_csv_path, index=False)
        # Extract the top 10 features from the reduced DataFrame
        top10_feature_names = top10_features['Feature'].tolist()

        # Include the 'condition' column along with the top 10 features
        columns_to_include = top10_feature_names + ['condition']
        top10_df = reduced_df[columns_to_include]

        # Save the extracted top 10 features to a CSV file
        top10_features_csv_path = os.path.join(output_dir, f"{model_name}_top10_features_data.csv")
        top10_df.to_csv(top10_features_csv_path, index=False)

        return {
            "message": "Top 10 feature importance computed and saved successfully.",
            "files": {
                "top10_features_txt": top10_txt_path,
                "top10_features_plot": plot_path,
                "top10_features_csv": top10_csv_path
            }
        }
    
        

    except Exception as e:
        return {"message": "Error during feature importance computation.", "error": str(e)}

if __name__ == "__main__":
    import sys
    model_name = sys.argv[1]
    input_file = sys.argv[2]
    output_dir = sys.argv[3]
    result = get_model_and_importance_with_top10(model_name, input_file, output_dir)
    print(result)
