import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
import os

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

def get_model_and_importance_with_top10(model_name, input_file, output_dir):
    try:
        # Sanitize model_name to avoid issues with spaces in file paths
        sanitized_model_name = model_name.replace(" ", "_")

        # Load data
        df = pd.read_csv(input_file)

        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Prepare features and target
        X = df.drop(columns=['condition'])
        y = df['condition']

        # Ensure the selected model exists
        if model_name not in classifiers:
            return {
                "message": f"Model '{model_name}' not found in classifiers.",
                "error": "Invalid model selection"
            }

        model = classifiers[model_name]

        # Train the model
        model.fit(X, y)

        # Compute feature importance
        if hasattr(model, "feature_importances_"):
            # Tree-based models
            importance_scores = model.feature_importances_
            importance_df = pd.DataFrame({
                'Feature': X.columns,
                'Importance': importance_scores
            }).sort_values(by='Importance', ascending=False)

        elif hasattr(model, "coef_"):
            # Linear models
            coef_scores = model.coef_.flatten()
            importance_df = pd.DataFrame({
                'Feature': X.columns,
                'Importance': coef_scores
            }).sort_values(by='Importance', ascending=False)

        else:
            return {
                "message": f"Model '{model_name}' does not support feature importance computation.",
                "error": "Feature importance not available"
            }

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

        # Extract the top 10 features from the DataFrame
        top10_feature_names = top10_features['Feature'].tolist()

        # Include the 'condition' column along with the top 10 features
        columns_to_include = top10_feature_names + ['condition']
        top10_df = df[columns_to_include]

        # Save the extracted top 10 features to a CSV file
        top10_features_csv_path = os.path.join(output_dir, f"{model_name}_top10_features_data.csv")
        top10_df.to_csv(top10_features_csv_path, index=False)

        return {
            "message": "Top 10 feature importance computed and saved successfully.",
            "files": {
                "top10_features_txt": top10_txt_path,
                "top10_features_plot": plot_path,
                "top10_features_csv": top10_csv_path,
                "top10_features_data_csv": top10_features_csv_path
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
