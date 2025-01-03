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
def visualize_dimensionality_reduction(input_file, output_dir,model_name):
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
        tsne_png = os.path.join(output_dir, f"{model_name}_tSNE_plot.png")
        tsne_pdf = os.path.join(output_dir, f"{model_name}_tSNE_plot.pdf")
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
        umap_png = os.path.join(output_dir, f"{model_name}_UMAP_plot.png")
        umap_pdf = os.path.join(output_dir, f"{model_name}_UMAP_plot.pdf")
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
        combined_png = os.path.join(output_dir, f"{model_name}_dimensionality_reduction_combined_{timestamp}.png")
        combined_pdf = os.path.join(output_dir, f"{model_name}_dimensionality_reduction_combined_{timestamp}.pdf")
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

if __name__ == "__main__":
    import sys

    # Define the input file and output directory
    input_file = sys.argv[1]
    output_dir = sys.argv[2]
    model_name = sys.argv[3]

    # Process the file
    result = visualize_dimensionality_reduction(input_file, output_dir,model_name)
    print(result)
