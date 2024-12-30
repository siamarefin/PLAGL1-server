import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import json

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

if __name__ == "__main__":
    import sys
    import json

    # Capture command-line arguments
    input_file = sys.argv[1]
    output_dir = sys.argv[2]
    drop_column = sys.argv[3]

    # Execute the function
    result = plot_correlation_clustermap(input_file, output_dir, drop_column)

    # Print the result as JSON
    print(json.dumps(result))
