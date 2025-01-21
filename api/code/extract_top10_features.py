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


def load_model_from_json(model_name, json_path):
    """
    Load model parameters from a JSON file and return an instantiated model.
    """
    # Map of model names to classes
    model_mapping = {
        "LogisticRegression": LogisticRegression,
        "ExtraTreesClassifier": ExtraTreesClassifier,
        "RandomForestClassifier": RandomForestClassifier,
        "XGBClassifier": XGBClassifier,
        "GradientBoostingClassifier": GradientBoostingClassifier,
        "SVC": SVC,
        "LinearDiscriminantAnalysis": LinearDiscriminantAnalysis,
        "QuadraticDiscriminantAnalysis": QuadraticDiscriminantAnalysis,
        "GaussianNB": GaussianNB,
        "KNeighborsClassifier": KNeighborsClassifier,
        "AdaBoostClassifier": AdaBoostClassifier,
    }

    try:
        # Load the JSON file
        with open(json_path, "r") as file:
            best_models = json.load(file)

        # Check if the model exists in the JSON
        if model_name not in best_models:
            raise ValueError(f"Model '{model_name}' not found in the JSON file.")

        # Extract model parameters
        model_info = best_models[model_name]
        model_class_name = model_info["model_name"]
        model_params = model_info["parameters"]

        # Validate and instantiate the model
        if model_class_name not in model_mapping:
            raise ValueError(f"Model class '{model_class_name}' is not supported.")

        model_class = model_mapping[model_class_name]
        return model_class(**model_params)

    except FileNotFoundError:
        raise ValueError(f"JSON file '{json_path}' not found.")
    except Exception as e:
        raise ValueError(f"Error loading model from JSON: {str(e)}")



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
            "LogisticRegression": LogisticRegression,
            "ExtraTreesClassifier": ExtraTreesClassifier,
            "RandomForestClassifier": RandomForestClassifier,
            "XGBClassifier": XGBClassifier,
            "GradientBoostingClassifier": GradientBoostingClassifier,
            "SVC": SVC,
            "LinearDiscriminantAnalysis": LinearDiscriminantAnalysis,
            "QuadraticDiscriminantAnalysis": QuadraticDiscriminantAnalysis,
            "GaussianNB": GaussianNB,
            "KNeighborsClassifier": KNeighborsClassifier,
            "AdaBoostClassifier": AdaBoostClassifier,
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


# if __name__ == "__main__":
#     try:
#         # Command-line arguments
#         model_name_arg = sys.argv[1]
#         input_file_arg = sys.argv[2]
#         output_dir_arg = sys.argv[3]
#         model_params  = sys.argv[4]

#         # print("json_input:", json_input)

#         # # Check if the JSON input is a string or a file path
#         # try:
#         #     best_models = json.loads(json_input)  # Try to parse the JSON string directly
#         #     print("best_models:",best_models)
#         #     print("Parsed input as a JSON string.")
#         # except json.JSONDecodeError:
#         #     with open(json_input, "r") as file:  # Otherwise, treat it as a file path
#         #         best_models = json.load(file)
#         #     print("Loaded input from a JSON file.")

#         # # Map user-friendly model names to their class names
#         # name_mapping = {
#         #     "Extra Trees": "ExtraTreesClassifier",
#         #     "Random Forest": "RandomForestClassifier",
#         #     "XGBoost": "XGBClassifier",
#         #     "Gradient Boosting": "GradientBoostingClassifier",
#         #     "Logistic Regression": "LogisticRegression",
#         #     "SVM": "SVC",
#         #     "LDA": "LinearDiscriminantAnalysis",
#         #     "QDA": "QuadraticDiscriminantAnalysis",
#         #     "Naive Bayes": "GaussianNB",
#         #     "K Neighbors": "KNeighborsClassifier",
#         #     "AdaBoost": "AdaBoostClassifier",
#         # }

#         # Resolve the model name
#         # model_name_resolved = name_mapping.get(model_name_arg, model_name_arg)

#         # print("model_name_resolved:",model_name_resolved)

#         # print(" ")
#         # print("best_models: ", best_models )

#         # Load the model parameters from the JSON data
#         # if model_name_resolved not in best_models.model_name:
#         #     raise ValueError(f"Model '{model_name_arg}' not found in the JSON data.")

#         # model_info = best_models[model_name_resolved]
#         # model_name = model_info["model_name"]
#         # model_params = model_info["parameters"]

#         # # Debugging: Print the loaded model and parameters
#         # print(f"Loaded model: {model_name}")
#         # print(f"Model parameters: {model_params}")

#         # Run the function with the loaded model
#         result = get_model_and_importance_with_top10(model_name_arg, model_params, input_file_arg, output_dir_arg)
#         print(json.dumps(result, indent=4))

#     except Exception as e:
#         print(json.dumps({"message": "An unexpected error occurred.", "error": str(e)}, indent=4))
