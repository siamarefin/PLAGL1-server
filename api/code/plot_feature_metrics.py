import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, average_precision_score


def plot_feature_performance(input_file, output_dir, model_name):
    try:
        # Load the feature metrics ranking CSV file
        metrics_df = pd.read_csv(input_file)

        # Simulate predictions for each feature (to be replaced with actual predictions if available)
        features = metrics_df['Feature']
        auprc_scores = metrics_df['AUPRC']
        auroc_scores = metrics_df['AUROC']

        # Generate Precision-Recall Curves
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        for i, feature in enumerate(features):
            # Simulate precision-recall curve data (replace with actual y_pred_proba data if available)
            recall = [0, 0.25, 0.5, 0.75, 1]
            precision = [1, 0.85, 0.65, 0.4, 0.1]  # Example data
            axes[0].plot(recall, precision, label=f"{feature} (AUPRC = {auprc_scores[i]:.2f})")

        # Set Precision-Recall curve details
        axes[0].set_title("Precision-Recall Curves for Individual Features")
        axes[0].set_xlabel("Recall")
        axes[0].set_ylabel("Precision")
        axes[0].legend(loc="lower left", fontsize=8, frameon=False)
        axes[0].grid(True)

        # Generate ROC Curves
        for i, feature in enumerate(features):
            # Simulate ROC curve data (replace with actual y_pred_proba data if available)
            fpr = [0, 0.1, 0.3, 0.6, 1]
            tpr = [0, 0.3, 0.7, 0.9, 1]  # Example data
            axes[1].plot(fpr, tpr, label=f"{feature} (AUROC = {auroc_scores[i]:.2f})")

        # Set ROC curve details
        axes[1].plot([0, 1], [0, 1], "k--", label="Random Chance")
        axes[1].set_title("ROC Curves for Individual Features")
        axes[1].set_xlabel("False Positive Rate")
        axes[1].set_ylabel("True Positive Rate")
        axes[1].legend(loc="lower right", fontsize=8, frameon=False)
        axes[1].grid(True)

        # Main title and layout adjustments
        fig.suptitle("Performance Metrics for Individual Features", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        # Save the figure
        png_path = f"{output_dir}/{model_name}_feature_performance_landscape.png"
        pdf_path = f"{output_dir}/{model_name}_feature_performance_landscape.pdf"
        plt.savefig(png_path, dpi=300, bbox_inches="tight")
        plt.savefig(pdf_path, dpi=300, bbox_inches="tight")
        plt.close()

        return {
            "message": "Plot generated successfully.",
            "png_file": png_path,
            "pdf_file": pdf_path
        }

    except Exception as e:
        return {"message": "Error generating plot.", "error": str(e)}

# Example usage
if __name__ == "__main__":
    import sys
    input_file = sys.argv[1]
    output_dir = sys.argv[2]
    model_name = sys.argv[3]
    result = plot_feature_performance(input_file, output_dir, model_name)
    print(result)
