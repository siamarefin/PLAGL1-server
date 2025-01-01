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

if __name__ == "__main__":
    import sys
    input_file = sys.argv[1]
    output_dir = sys.argv[2]
    feature_ratio = float(sys.argv[3])  # Pass feature ratio dynamically
    result = feature_selection_and_model(input_file, output_dir, feature_ratio)
    print(result)
