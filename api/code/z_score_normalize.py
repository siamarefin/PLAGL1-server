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


if __name__ == "__main__":
    import sys
    # Define the input file and output directory
    input_file = sys.argv[1]
    output_dir = sys.argv[2]
    
    # Process the file
    result = process_file(input_file, output_dir)
    print(result)
