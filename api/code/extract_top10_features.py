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
import sys
import json


def get_model_and_importance_with_top10(model_name, model_params, input_file, output_dir):
    try:
        # Validate model_params
        if not isinstance(model_params, dict):
            raise ValueError(f"model_params must be a dictionary, got {type(model_params).__name__}")

        # Sanitize model_name to avoid issues with spaces in file paths
        sanitized_model_name = model_name.replace(" ", "_")

        # Load data
        df = pd.read_csv(input_file)

        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Prepare features and target
        X = df.drop(columns=["condition"])
        y = df["condition"]

        # Map of model names to classes
        model_mapping = {
            "Logistic Regression": LogisticRegression,
            "Extra Trees": ExtraTreesClassifier,
            "Random Forest": RandomForestClassifier,
            "XGBoost": XGBClassifier,
            "Gradient Boosting": GradientBoostingClassifier,
            "SVM": SVC,
            "LDA": LinearDiscriminantAnalysis,
            "QDA": QuadraticDiscriminantAnalysis,
            "Naive Bayes": GaussianNB,
            "K Neighbors": KNeighborsClassifier,
            "AdaBoost": AdaBoostClassifier,
        }

        # Validate and instantiate the model
        if model_name not in model_mapping:
            return {
                "message": f"Model '{model_name}' not supported.",
                "error": "Invalid model selection",
            }

        model = model_mapping[model_name](**model_params)  # Dynamically instantiate the model

        # Train the model
        model.fit(X, y)

        # Compute feature importance
        if hasattr(model, "feature_importances_"):
            # Tree-based models
            importance_scores = model.feature_importances_
            importance_df = pd.DataFrame(
                {"Feature": X.columns, "Importance": importance_scores}
            ).sort_values(by="Importance", ascending=False)

        elif hasattr(model, "coef_"):
            # Linear models
            coef_scores = model.coef_.flatten()
            importance_df = pd.DataFrame(
                {"Feature": X.columns, "Importance": coef_scores}
            ).sort_values(by="Importance", ascending=False)

        else:
            return {
                "message": f"Model '{model_name}' does not support feature importance computation.",
                "error": "Feature importance not available",
            }

        # Select top 10 features
        top10_features = importance_df.head(10)

        # Save top 10 feature names to a text file
        top10_txt_path = os.path.join(output_dir, f"{sanitized_model_name}_top10_features.txt")
        with open(top10_txt_path, "w") as f:
            f.write("Top 10 Feature Importance Scores:\n")
            f.write(top10_features.to_string(index=False))

        # Plot top 10 feature importance
        plt.figure(figsize=(8, 5))
        plt.barh(
            top10_features["Feature"],
            top10_features["Importance"],
            color="skyblue",
            align="center",
        )
        plt.gca().invert_yaxis()
        plt.title(f"Top 10 Feature Importance for {model_name}")
        plt.xlabel("Importance Score")
        plt.ylabel("Feature")
        plt.tight_layout()

        # Save the plot
        plot_path = os.path.join(output_dir, f"{sanitized_model_name}_top10_features.png")
        plt.savefig(plot_path)
        plt.close()

        # Save top 10 features DataFrame to a CSV file
        top10_csv_path = os.path.join(output_dir, f"{sanitized_model_name}_top10_features.csv")
        top10_features.to_csv(top10_csv_path, index=False)

        return {
            "message": "Top 10 feature importance computed and saved successfully.",
            "files": {
                "top10_features_txt": top10_txt_path,
                "top10_features_plot": plot_path,
                "top10_features_csv": top10_csv_path,
            },
        }

    except Exception as e:
        return {"message": "Error during feature importance computation.", "error": str(e)}


if __name__ == "__main__":
    try:
        model_name = sys.argv[1]
        input_file = sys.argv[2]
        output_dir = sys.argv[3]
        model_params = json.loads(sys.argv[4])  # Parse JSON string to dictionary
    except json.JSONDecodeError as e:
        print(json.dumps({"message": "Error parsing model parameters.", "error": str(e)}, indent=4))
        sys.exit(1)

    # Debugging: Print type and content of model_params
    print(f"Type of model_params: {type(model_params)}")  # Should print <class 'dict'>
    print(f"Contents of model_params: {model_params}")

