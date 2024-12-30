import os
import pandas as pd
import json  # Import json to format output properly

# Define the dataset path (modify as needed)
dataset_path = os.path.join("code", "1", "files", "count_data.csv")

try:
    # Check if the file exists
    if not os.path.exists(dataset_path):
        result = {"success": False, "message": f"File not found: {dataset_path}"}
    else:
        # Read the dataset
        df = pd.read_csv(dataset_path)

        # Extract column names
        column_names = list(df.columns)
        result = {"success": True, "columns": column_names}

    # Print the result as JSON
    print(json.dumps(result))  # Properly format the output as JSON

except Exception as e:
    error_result = {"success": False, "message": str(e)}
    print(json.dumps(error_result))  # Format error output as JSON
