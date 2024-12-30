import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split, cross_val_score
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
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

        # Apply RFE with RF
        rf_model = RandomForestClassifier(random_state=123)
        num_features_to_select = int(X_train.shape[1] * feature_ratio)
        rfe = RFE(estimator=rf_model, n_features_to_select=num_features_to_select, step=1)
        rfe.fit(X_train, y_train)

        # Select features
        selected_features = X_train.columns[rfe.support_]
        X_selected = df[selected_features]  # Reduced dataset with selected features
        X_selected['condition'] = df['condition']  # Add condition column back

        # Save selected features list
        selected_features_path = os.path.join(output_dir, "selected_features.csv")
        X_selected.to_csv(selected_features_path, index=False)

        # Train and evaluate
        X_train_reduced = X_train[selected_features]
        X_test_reduced = X_test[selected_features]
        rf_model_reduced = RandomForestClassifier(random_state=123)
        rf_model_reduced.fit(X_train_reduced, y_train)
        cv_scores = cross_val_score(rf_model_reduced, X_train_reduced, y_train, cv=5, scoring='roc_auc')
        y_pred_proba = rf_model_reduced.predict_proba(X_test_reduced)[:, 1]
        test_auc = roc_auc_score(y_test, y_pred_proba)

        # Return results
        result = {
            "message": "Feature selection and model training completed successfully.",
            "output_files": {
                "selected_features_csv": selected_features_path
            },
            "model_metrics": {
                "cross_validation_auc": cv_scores.mean(),
                "test_auc": test_auc
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
